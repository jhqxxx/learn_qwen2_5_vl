use std::{thread, time};

use candle_core::{DType, Error, IndexOp, Result, Tensor, D};
use candle_nn::{Activation, Init, Linear, Module, RmsNorm, VarBuilder, linear, rms_norm};

use crate::{
    config::Qwen2_5VLVisionConfig, qwen2_5_vl::utils::safe_arg_sort_last_dim, rope::{apply_rotary_pos_emb_vision, Qwen2_5VisionRotaryEmbedding}
};

pub struct Qwen2_5VisionPatchEmbed {
    patch_size: usize,
    temporal_patch_size: usize,
    in_channels: usize,
    embed_dim: usize,
    conv3d_weight: Tensor,
}

impl Qwen2_5VisionPatchEmbed {
    pub fn new(cfg: &Qwen2_5VLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let patch_size = cfg.patch_size;
        let temporal_patch_size = cfg.temporal_patch_size;
        let in_channels = cfg.in_channels;
        let embed_dim = cfg.hidden_size;
        // conv3d weight key: visual.patch_embed.proj.weight, value: Tensor[dims 1280, 3, 2, 14, 14; bf16, cuda:0]
        let conv3d_weight = vb.get_with_hints(
            (
                embed_dim,
                in_channels,
                temporal_patch_size,
                patch_size,
                patch_size,
            ),
            "proj.weight",
            Init::Const(1.),
        )?;
        Ok(Self {
            patch_size,
            temporal_patch_size,
            in_channels,
            embed_dim,
            conv3d_weight,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // hidden_states shape:  (grid_t*grid_h*grid_w, c*temporal_patch_size*patch_size*patch_size)
        // (seq_len, hidden_size) -> ((), 3, 2, 14, 14)
        let hidden_states = hidden_states.reshape((
            (),
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        ))?;
        // ((), 3, 2, 14, 14) -> ((), 1176)
        let hidden_states = hidden_states.flatten(1, 4)?;
        // (1280, 3, 2, 14, 14) -> (1280, 1176) -> (1176, 1280)
        let conv3d_weight = self.conv3d_weight.flatten(1, 4)?.t()?;
        // ((), 1176) matmul (1176, 1280) -> ((), 1280)
        let hidden_states = hidden_states.matmul(&conv3d_weight)?;
        Ok(hidden_states)
    }
}

pub struct Qwen2_5VLPatchMerger {
    hidden_size: usize,
    ln_q: RmsNorm,
    mlp_0: Linear,
    mlp_2: Linear,
}

impl Qwen2_5VLPatchMerger {
    pub fn new(cfg: &Qwen2_5VLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size * (cfg.spatial_merge_size.pow(2));
        let ln_q = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("ln_q"))?;
        let mlp_0 = linear(hidden_size, hidden_size, vb.pp("mlp.0"))?;
        let mlp_2 = linear(hidden_size, cfg.out_hidden_size, vb.pp("mlp.2"))?;
        Ok(Self {
            hidden_size,
            ln_q,
            mlp_0,
            mlp_2,
        })
    }
}
impl Module for Qwen2_5VLPatchMerger {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.apply(&self.ln_q)?.reshape(((), self.hidden_size))?;
        let xs = xs.apply(&self.mlp_0)?.gelu()?.apply(&self.mlp_2)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct Qwen2_5VLVisionMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl Qwen2_5VLVisionMLP {
    fn new(cfg: &Qwen2_5VLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = linear(hidden_sz, intermediate_sz, vb.pp("gate_proj"))?;
        let up_proj = linear(hidden_sz, intermediate_sz, vb.pp("up_proj"))?;
        let down_proj = linear(intermediate_sz, hidden_sz, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for Qwen2_5VLVisionMLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct Qwen2_5VLVisionAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    scale: Tensor,
}

impl Qwen2_5VLVisionAttention {
    fn new(cfg: &Qwen2_5VLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let num_heads = cfg.num_heads;
        let head_dim = hidden_size / num_heads;
        let qkv = linear(hidden_size, hidden_size * 3, vb.pp("qkv"))?;
        let proj = linear(hidden_size, hidden_size, vb.pp("proj"))?;
        let scale = Tensor::new(vec![1f32 / (head_dim as f32).sqrt()], vb.device())?.to_dtype(vb.dtype())?;
        Ok(Self {
            qkv,
            proj,
            num_heads,
            head_dim,
            hidden_size,
            scale
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        cu_seqlens: &Tensor,
    ) -> Result<Tensor> {
        // xs: (seq_len, hidden_size)
        let seq_length = xs.dim(0)?;
        // (seq_len, hidden_size) -> (seq_len, hidden_size*3)
        // -> (seq_len, 3, num_heads, head_dim) -> (3, seq_len, num_heads, head_dim)
        let qkv_states = xs
            .apply(&self.qkv)?
            .reshape((seq_length, 3, self.num_heads, ()))?
            .permute((1, 0, 2, 3))?;
        // (seq_len, num_heads, head_dim)
        let query_states = qkv_states.i(0)?.contiguous()?;
        let key_states = qkv_states.i(1)?.contiguous()?;
        let value_states = qkv_states.i(2)?.contiguous()?;
        let (query_states, key_states) =
            apply_rotary_pos_emb_vision(&query_states, &key_states, cos, sin)?;
        // (seq_len, num_heads, head_dim) -> (num_heads, seq_len, head_dim)
        let query_states = query_states.transpose(0, 1)?.contiguous()?;
        let key_states = key_states.transpose(0, 1)?.contiguous()?;
        let value_states = value_states.transpose(0, 1)?.contiguous()?;

        let mut attention_mask = Tensor::new(f32::NEG_INFINITY, query_states.device())?
            .broadcast_as((1, seq_length, seq_length))?;
        for i in 1..cu_seqlens.dim(0)? {
            let start = cu_seqlens.i(i - 1)?.to_scalar::<u32>()? as usize;
            let end = cu_seqlens.i(i)?.to_scalar::<u32>()? as usize;
            let block_size = end - start;
            let ones = Tensor::ones(
                (1, block_size, block_size),
                candle_core::DType::F32,
                query_states.device(),
            )?;
            attention_mask =
                attention_mask.slice_assign(&[(0..1), (start..end), (start..end)], &ones)?;
        }
        let attention_mask = attention_mask.to_dtype(query_states.dtype())?.contiguous()?;
        let attn_output = {
            let attn_weights = query_states.matmul(&key_states.transpose(D::Minus2, D::Minus1)?)?.broadcast_mul(&self.scale)?;
            let attn_weights = attn_weights.broadcast_add(&attention_mask)?;
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            attn_weights.matmul(&value_states)?
        };
        // (num_heads, seq_len, head_dim) -> (seq_len, num_heads, head_dim) -> (seq_len, hidden_size)
        let attn_output = attn_output.transpose(0, 1)?.reshape((seq_length, ()))?.contiguous()?;
        attn_output.apply(&self.proj)
    }
}

#[derive(Debug, Clone)]
struct Qwen2_5VLVisionBlock {
    attn: Qwen2_5VLVisionAttention,
    mlp: Qwen2_5VLVisionMLP,
    norm1: RmsNorm,
    norm2: RmsNorm,
}

impl Qwen2_5VLVisionBlock {
    fn new(cfg: &Qwen2_5VLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let attn = Qwen2_5VLVisionAttention::new(cfg, vb.pp("attn"))?;
        let mlp = Qwen2_5VLVisionMLP::new(cfg, vb.pp("mlp"))?;
        let norm1 = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm1"))?;
        let norm2 = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm2"))?;

        Ok(Self {
            attn,
            mlp,
            norm1,
            norm2,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        cu_seqlens: &Tensor,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.norm1.forward(xs)?;
        let xs = self.attn.forward(&xs, cos, sin, cu_seqlens)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.norm2)?.apply(&self.mlp)?;
        residual + xs
    }
}

pub struct Qwen2_5_VisionTransformerPretrainedModel {
    spatial_merge_size: usize,
    patch_size: usize,
    fullatt_block_indexes: Vec<usize>,
    window_size: usize,
    spatial_merge_unit: usize,
    head_dim: usize,
    patch_embed: Qwen2_5VisionPatchEmbed,
    rotary_pos_emb: Qwen2_5VisionRotaryEmbedding,
    blocks: Vec<Qwen2_5VLVisionBlock>,
    merger: Qwen2_5VLPatchMerger,
    dtype: DType,
}

impl Qwen2_5_VisionTransformerPretrainedModel {
    pub fn new(cfg: &Qwen2_5VLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let spatial_merge_size = cfg.spatial_merge_size;
        let patch_size = cfg.patch_size;
        let fullatt_block_indexes = cfg.fullatt_block_indexes.clone();
        let window_size = cfg.window_size;
        let spatial_merge_unit = spatial_merge_size * spatial_merge_size;
        let head_dim = cfg.hidden_size / cfg.num_heads;
        let patch_embed = Qwen2_5VisionPatchEmbed::new(cfg, vb.pp("patch_embed"))?;
        let rotary_pos_emb = Qwen2_5VisionRotaryEmbedding::new(head_dim / 2, cfg.rope_theta);
        let mut blocks = Vec::new();
        let vb_blocks = vb.pp("blocks");
        for i in 0..cfg.depth {
            let block = Qwen2_5VLVisionBlock::new(cfg, vb_blocks.pp(i))?;
            blocks.push(block);
        }
        let merger = Qwen2_5VLPatchMerger::new(cfg, vb.pp("merger"))?;
        let dtype = vb.dtype();
        Ok(Self {
            spatial_merge_size,
            patch_size,
            fullatt_block_indexes,
            window_size,
            spatial_merge_unit,
            head_dim,
            patch_embed,
            rotary_pos_emb,
            blocks,
            merger,
            dtype
        })
    }

    pub fn rot_pos_emb(&self, grid_thw: &Tensor) -> Result<Tensor> {
        let mut pos_ids = Vec::new();
        for i in 0..grid_thw.dim(0)? {
            let [t, h, w] = grid_thw.i(i)?.to_vec1::<u32>()?[..] else {
                return Err(Error::Msg(format!("grid_thw Expected exactly 3 elements")));
            };
            let hpos_ids = Tensor::arange(0, h, grid_thw.device())?
                .unsqueeze(1)?
                .expand((h as usize, w as usize))?;
            let hpos_ids = hpos_ids.reshape((
                h as usize / self.spatial_merge_size,
                self.spatial_merge_size,
                w as usize / self.spatial_merge_size,
                self.spatial_merge_size,
            ))?;
            let hpos_ids = hpos_ids.permute((0, 2, 1, 3))?.flatten_all()?;

            let wpos_ids = Tensor::arange(0, w, grid_thw.device())?
                .unsqueeze(0)?
                .expand((h as usize, w as usize))?;
            let wpos_ids = wpos_ids.reshape((
                h as usize / self.spatial_merge_size,
                self.spatial_merge_size,
                w as usize / self.spatial_merge_size,
                self.spatial_merge_size,
            ))?;
            let wpos_ids = wpos_ids.permute((0, 2, 1, 3))?.flatten_all()?;
            let thw_pos_ids =
                Tensor::stack(&[&hpos_ids, &wpos_ids], D::Minus1)?.repeat((t as usize, 1))?;
            pos_ids.push(thw_pos_ids);
        }
        let pos_ids = Tensor::cat(&pos_ids, 0)?;
        let max_grid_size = grid_thw.i((.., 1..))?.max_all()?.to_scalar::<u32>()?;
        let rotary_pos_emb_full = self
            .rotary_pos_emb
            .forward(max_grid_size as usize, grid_thw.device())?;

        // contiguous()一定要加！！！很重要！！！！，不然index_select出来的是错的
        // 找错找了半天，都是泪啊，做维度索引操作后contiguous顺手写上总没错
        let pos_ids_0 = pos_ids.i((.., 0))?.contiguous()?;
        let pos_ids_1 = pos_ids.i((.., 1))?.contiguous()?;
        let rotary_pos_emb_0 = rotary_pos_emb_full.index_select(&pos_ids_0, 0)?;
        let rotary_pos_emb_1 = rotary_pos_emb_full.index_select(&pos_ids_1, 0)?;
        let rotary_pos_emb = Tensor::cat(&[rotary_pos_emb_0, rotary_pos_emb_1], 1)?.contiguous()?;
        Ok(rotary_pos_emb)
    }

    pub fn get_window_index(&self, grid_thw: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut window_index = Vec::new();
        let mut cu_window_seqlens = vec![0];
        let mut window_index_id = 0_i64;

        let vit_merger_window_size =
            (self.window_size / self.spatial_merge_size / self.patch_size) as u32;
        for i in 0..grid_thw.dim(0)? {
            let [grid_t, grid_h, grid_w] = grid_thw.i(i)?.to_vec1::<u32>()?[..] else {
                return Err(Error::Msg(format!("grid_thw Expected exactly 3 elements")));
            };
            let llm_grid_h = grid_h / self.spatial_merge_size as u32;
            let llm_grid_w = grid_w / self.spatial_merge_size as u32;
            // 因为后续需要使用-100来做填充，所以需要int类型
            // candle好像不支持i32， DType里面都没有定义i32, 所以这里使用i64
            let mut index = Tensor::arange(
                0_i64,
                (grid_t * llm_grid_h * llm_grid_w) as i64,
                grid_thw.device(),
            )?
            .reshape((grid_t as usize, llm_grid_h as usize, llm_grid_w as usize))?
            .contiguous()?;
            // python transformers 中实现如下
            // let pad_h = (vit_merger_window_size - llm_grid_h % vit_merger_window_size);
            // 后面加上 % vit_merger_window_size，
            // 保证llm_grid_h能整除vit_merger_window_size时不需要pad
            // 按理说能整除应该是不需要pad的，transformers中这样实现不知道是不是有什么其他原因
            let pad_h = (vit_merger_window_size - llm_grid_h % vit_merger_window_size)
                % vit_merger_window_size;
            let pad_w = (vit_merger_window_size - llm_grid_w % vit_merger_window_size)
                % vit_merger_window_size;
            let num_window_h = (llm_grid_h + pad_h) / vit_merger_window_size;
            let num_window_w = (llm_grid_w + pad_w) / vit_merger_window_size;
            if pad_h > 0 {
                let pad_h_t = Tensor::new(vec![-100_i64], grid_thw.device())?
                    .broadcast_as((grid_t as usize, pad_h as usize, llm_grid_w as usize))?
                    .contiguous()?;
                index = Tensor::cat(&[&index, &pad_h_t], 1)?;
            }
            if pad_w > 0 {
                let pad_w_t = Tensor::new(vec![-100_i64], grid_thw.device())?
                    .broadcast_as((
                        grid_t as usize,
                        (llm_grid_h + pad_h) as usize,
                        pad_w as usize,
                    ))?
                    .contiguous()?;
                index = Tensor::cat(&[&index, &pad_w_t], 2)?;
            }
            let index_padded = index
                .reshape((
                    grid_t as usize,
                    num_window_h as usize,
                    vit_merger_window_size as usize,
                    num_window_w as usize,
                    vit_merger_window_size as usize,
                ))?
                .permute((0, 1, 3, 2, 4))?;
            let index_padded = index_padded
                .reshape((
                    grid_t as usize,
                    (num_window_h * num_window_w) as usize,
                    vit_merger_window_size as usize,
                    vit_merger_window_size as usize,
                ))?
                .contiguous()?;
            let is_pad = Tensor::new(vec![-100_i64], grid_thw.device())?;
            let seqlens = index_padded
                .broadcast_ne(&is_pad)?
                .sum((2, 3))?
                .flatten_all()?;
            let index_padded = index_padded.flatten_all()?;
            let not_pad = index_padded.broadcast_ne(&is_pad)?.to_vec1::<u8>()?;
            let indices: Vec<u32> = not_pad
                .iter()
                .enumerate()
                .filter_map(|(idx, &val)| if val != 0 { Some(idx as u32) } else { None })
                .collect();
            let indices_tensor = Tensor::from_slice(&indices, indices.len(), grid_thw.device())?;
            let index_new = index_padded.gather(&indices_tensor, 0)?;
            let index_new =
                index_new.broadcast_add(&Tensor::new(vec![window_index_id], grid_thw.device())?)?;
            window_index.push(index_new);

            let seq_len_last = cu_window_seqlens[cu_window_seqlens.len() - 1];
            // cumsum方法i64类型执行会报错，先转成F64计算后再转回i64
            let cu_seqlens_tmp = seqlens
                .to_dtype(candle_core::DType::F64)?
                .cumsum(0)?
                .to_dtype(candle_core::DType::I64)?
                .broadcast_mul(&Tensor::new(
                    vec![self.spatial_merge_unit as i64],
                    grid_thw.device(),
                )?)?
                .broadcast_add(&Tensor::new(vec![seq_len_last], grid_thw.device())?)?;
            cu_window_seqlens.extend_from_slice(&cu_seqlens_tmp.to_vec1::<i64>()?);
            window_index_id += (grid_t * llm_grid_h * llm_grid_w) as i64;
        }
        let window_index_tensor = Tensor::cat(&window_index, 0)?;
        let cu_window_seqlens_tensor = Tensor::from_slice(
            &cu_window_seqlens,
            cu_window_seqlens.len(),
            grid_thw.device(),
        )?.to_dtype(candle_core::DType::U32)?;
        Ok((window_index_tensor, cu_window_seqlens_tensor))
    }

    pub fn forward(&self, hidden_states: &Tensor, grid_thw: &Tensor) -> Result<Tensor> {
        // hidden_states: (seq_len, hidden_size)
        // grid_thw: (num_images_or_videos, 3), temporal, height, width
        let hidden_states = hidden_states.to_dtype(self.dtype)?;
        let hidden_states = self.patch_embed.forward(&hidden_states)?;
        let rotary_pos_emb = self.rot_pos_emb(grid_thw)?;
        let (window_index, cu_window_seqlens) = self.get_window_index(grid_thw)?;
        let seq_len = hidden_states.dim(0)?;
        let hidden_states = hidden_states
            .reshape((
                seq_len / self.spatial_merge_unit,
                self.spatial_merge_unit,
                (),
            ))?
            .contiguous()?;
        let hidden_states = hidden_states.index_select(&window_index, 0)?;
        let mut hidden_states = hidden_states.reshape((seq_len, ()))?;
        let rotary_pos_emb = rotary_pos_emb.reshape((
            seq_len / self.spatial_merge_unit,
            self.spatial_merge_unit,
            (),
        ))?;
        let rotary_pos_emb = rotary_pos_emb.index_select(&window_index, 0)?;
        let rotary_pos_emb = rotary_pos_emb.reshape((seq_len, ()))?;
        let emb = Tensor::cat(&[&rotary_pos_emb, &rotary_pos_emb], D::Minus1)?;
        let cos = emb.cos()?.to_dtype(hidden_states.dtype())?;
        let sin = emb.sin()?.to_dtype(hidden_states.dtype())?;
        let cu_seqlens = grid_thw.i((.., 1))?.mul(&grid_thw.i((.., 2))?)?;
        let grid_t = grid_thw.i((.., 0))?.to_vec1::<u32>()?;
        let cu_seqlens_repeat: Vec<Tensor> = (0..cu_seqlens.dim(0)?)
            .map(|i| cu_seqlens.i(i)?.repeat(grid_t[i] as usize))
            .collect::<Result<Vec<_>>>()?;
        let cu_seqlens = Tensor::cat(&cu_seqlens_repeat, 0)?;
        let pad_zero = Tensor::from_vec(vec![0_u32], cu_seqlens.dim(0)?, hidden_states.device())?;
        let cu_seqlens = Tensor::cat(&[&pad_zero, &cu_seqlens], D::Minus1)?;
        let mut cu_seqlens_now = cu_seqlens.clone();
        for (layer_num, block) in self.blocks.iter().enumerate() {
            
            if self.fullatt_block_indexes.contains(&layer_num) {
                cu_seqlens_now = cu_seqlens.clone();
            } else {
                cu_seqlens_now = cu_window_seqlens.clone();
            }
            hidden_states = block.forward(&hidden_states, &cos, &sin, &cu_seqlens_now)?;
        }
        let hidden_states = self.merger.forward(&hidden_states)?;
        let reverse_indices = safe_arg_sort_last_dim(&window_index, true)?;
        let hidden_states = hidden_states.index_select(&reverse_indices, 0)?;
        Ok(hidden_states)
    }
}
