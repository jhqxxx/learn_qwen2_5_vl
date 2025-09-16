use crate::rope::{Qwen2_5VLTextRotaryEmbedding, apply_multimodel_rotary_pos_emb};
use crate::{config::RopeScaling, qwen2_5_vl::config::Qwen2_5VLTextConfig};
use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::{
    Activation, Linear, Module, RmsNorm, VarBuilder, linear, linear_no_bias, rms_norm,
};
pub fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(xs)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
        // Using cat is faster than a broadcast as it avoids going through a potentially
        // strided copy.
        // https://github.com/huggingface/candle/pull/2043
        Tensor::cat(&vec![&xs; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}

#[derive(Debug, Clone)]
struct Qwen2_5VLTextMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl Qwen2_5VLTextMLP {
    fn new(cfg: &Qwen2_5VLTextConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_sz, hidden_sz, vb.pp("down_proj"))?;
        
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for Qwen2_5VLTextMLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct Qwen2_5VLTextAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    kv_cache: Option<(Tensor, Tensor)>,
    rope_scaling: RopeScaling,
}

impl Qwen2_5VLTextAttention {
    fn new(cfg: &Qwen2_5VLTextConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = hidden_size / num_heads;
        let rope_scaling = cfg.rope_scaling.clone();
        let q_proj = linear(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(hidden_size, hidden_size, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size,
            kv_cache: None,
            rope_scaling,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;
        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let (query_states, key_states) = apply_multimodel_rotary_pos_emb(
            &query_states,
            &key_states,
            cos,
            sin,
            self.rope_scaling.mrope_section.clone(),
        )?;
        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };

        self.kv_cache = Some((key_states.clone(), value_states.clone()));

        let key_states = repeat_kv(key_states, self.num_kv_groups)?.contiguous()?;
        let value_states = repeat_kv(value_states, self.num_kv_groups)?.contiguous()?;
        let query_states = query_states.contiguous()?;
        let attn_output = {
            let attn_weights = query_states.matmul(&key_states.transpose(D::Minus2, D::Minus1)?)?;
            let scale = 1f64 / f64::sqrt(self.head_dim as f64);
            let attn_weights = (attn_weights * scale)?;
            let attn_weights = match attention_mask {
                None => attn_weights,
                Some(mask) => attn_weights.broadcast_add(mask)?,
            };
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            let attn_weights = attn_weights.matmul(&value_states)?;
            attn_weights
        };
        let attn_output =
            attn_output
                .transpose(1, 2)?
                .contiguous()?
                .reshape((b_sz, q_len, self.hidden_size))?;
        let attn_output = attn_output.apply(&self.o_proj)?;
        Ok(attn_output)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }
}

#[derive(Debug, Clone)]
struct Qwen2_5VLTextDecoderLayer {
    self_attn: Qwen2_5VLTextAttention,
    mlp: Qwen2_5VLTextMLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Qwen2_5VLTextDecoderLayer {
    fn new(cfg: &Qwen2_5VLTextConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn = Qwen2_5VLTextAttention::new(cfg, vb.pp("self_attn"))?;
        let mlp = Qwen2_5VLTextMLP::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, cos, sin, attention_mask)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        residual + xs
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache()
    }
}

#[derive(Debug, Clone)]
pub struct Qwen2_5VLTextModel {
    pub embed_tokens: candle_nn::Embedding,
    layers: Vec<Qwen2_5VLTextDecoderLayer>,
    norm: RmsNorm,
    rotary_emb: Qwen2_5VLTextRotaryEmbedding,
    dtype: DType,
    sliding_window: usize,
    device: Device,
}

impl Qwen2_5VLTextModel {
    pub fn new(cfg: &Qwen2_5VLTextConfig, vb: VarBuilder) -> Result<Self> {
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let rotary_emb = Qwen2_5VLTextRotaryEmbedding::new(cfg.head_dim, cfg.rope_theta);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = Qwen2_5VLTextDecoderLayer::new(cfg, vb_l.pp(layer_idx))?;
            layers.push(layer)
        }
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        let sliding_window = cfg.sliding_window;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_emb,
            dtype: vb.dtype(),
            sliding_window,
            device: vb.device().clone()
        })
    }

    fn prepare_causal_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        // Sliding window mask?
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || j + self.sliding_window < i {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), self.dtype, &self.device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
            .to_dtype(self.dtype)
    }
    pub fn forward(
        &mut self,
        inputs_embeds: &Tensor,
        seqlen_offset: usize,
        position_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_size, seq_len, _) = inputs_embeds.dims3()?;

        let position_ids = match position_ids {
            Some(ids) => ids.clone(),
            None => Tensor::arange(
                seqlen_offset as u32,
                (seq_len + seqlen_offset) as u32,
                inputs_embeds.device(),
            )?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as((3, b_size, seq_len))?,
        };
        let (cos, sin) = self.rotary_emb.forward(&position_ids, self.dtype)?;
        let mut xs = inputs_embeds.clone();
        let attention_mask: Option<&Tensor> = {
            if seq_len <= 1 {
                None
            } else {
                Some(&self.prepare_causal_attention_mask(b_size, seq_len, seqlen_offset)?)
            }
        };
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, &cos, &sin, attention_mask)?; 
        }
        let xs = xs.apply(&self.norm)?;
        Ok(xs)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }
}
