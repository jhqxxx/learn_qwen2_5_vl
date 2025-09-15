use crate::{
    config::Qwen2_5VLConfig,
    qwen2_5_vl::{
        processor::GeneralInput,
        text_model::Qwen2_5VLTextModel,
        utils::{
            get_equal_mask, get_vision_next_indices, masked_scatter_dim0, nonzero_index,
            nonzero_slice, zero_index,
        },
        vision_model::Qwen2_5VisionTransformerPretrainedModel,
    },
};
use candle_core::{D, Error, IndexOp, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};

pub struct Qwen2_5_VLForConditionalGeneration {
    visual: Qwen2_5VisionTransformerPretrainedModel,
    model: Qwen2_5VLTextModel,
    cfg: Qwen2_5VLConfig,
    lm_head: Linear,
    rope_deltas: Option<Tensor>,
}

impl Qwen2_5_VLForConditionalGeneration {
    pub fn new(cfg: Qwen2_5VLConfig, vb: VarBuilder) -> Result<Self> {
        let visual =
            Qwen2_5VisionTransformerPretrainedModel::new(&cfg.vision_config, vb.pp("visual"))?;
        let model = Qwen2_5VLTextModel::new(&cfg.text_config, vb.pp("model"))?;
        let vocab_size = cfg.text_config.vocab_size;
        let lm_head = if cfg.text_config.tie_word_embeddings {
            Linear::new(model.embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.text_config.hidden_size, vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self {
            visual,
            model,
            cfg,
            lm_head,
            rope_deltas: None,
        })
    }

    pub fn get_rope_index(
        &self,
        input_ids: &Tensor,
        image_grid_thw: Option<&Tensor>,
        video_grid_thw: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let spatial_merge_size = self.cfg.vision_config.spatial_merge_size;
        let mut mrope_position_deltas: Vec<i64> = Vec::new();
        if image_grid_thw.is_some() || video_grid_thw.is_some() {
            let total_input_ids = input_ids.clone();
            let mut mask_;
            if mask.is_none() {
                mask_ = Tensor::ones_like(&total_input_ids)?;
            } else {
                mask_ = mask.unwrap().clone();
            }
            let mut position_ids = Tensor::ones(
                (3, input_ids.dim(0)?, input_ids.dim(1)?),
                input_ids.dtype(),
                input_ids.device(),
            )?;
            let mut image_index = 0;
            let mut video_index = 0;
            for i in 0..total_input_ids.dim(0)? {
                let mut input_ids_i = total_input_ids.i(i)?;
                let mask_i = mask_.i(i)?;
                // 推理时, attention_mask如果是全1向量,取非0索引的操作没必要
                if mask_i.sum_all()?.to_scalar::<u32>()? != mask_i.dim(0)? as u32 {
                    let nonzero_idx = nonzero_index(&mask_i)?;
                    input_ids_i = input_ids_i.gather(&nonzero_idx, 0)?;
                }
                let mut text_start = 0;
                let mut text_end = 0;
                let mut thw = vec![];
                let mut second_per_grid_t = 0u32;
                let mut llm_pos_ids_list: Vec<Tensor> = Vec::new();
                // vision start的下一个索引
                let vision_indices =
                    get_vision_next_indices(&input_ids_i, self.cfg.vision_start_token_id as u32);
                match vision_indices {
                    Ok(indeices) => {
                        let vision_tokens = input_ids_i.gather(&indeices, 0)?.to_vec1::<u32>()?;
                        let vision_indices_vec = indeices.to_vec1::<u32>()?;
                        for (j, &token) in vision_tokens.iter().enumerate() {
                            if token == self.cfg.image_token_id as u32 {
                                thw = image_grid_thw.unwrap().i(image_index)?.to_vec1::<u32>()?;
                                image_index += 1;
                                text_end = vision_indices_vec[j];
                                second_per_grid_t = 0;
                            }
                            if token == self.cfg.video_token_id as u32 {
                                thw = video_grid_thw.unwrap().i(video_index)?.to_vec1::<u32>()?;
                                video_index += 1;
                                text_end = vision_indices_vec[j];
                                second_per_grid_t = 1;
                            }
                            let llm_grid_t = thw[0];
                            let llm_grid_h = thw[1] / spatial_merge_size as u32;
                            let llm_grid_w = thw[2] / spatial_merge_size as u32;
                            let text_len = text_end - text_start;
                            let start_idx = if llm_pos_ids_list.len() > 0 {
                                llm_pos_ids_list[llm_pos_ids_list.len() - 1]
                                    .max_all()?
                                    .to_scalar::<u32>()?
                                    + 1
                            } else {
                                0
                            };
                            let pos_ids =
                                Tensor::arange(start_idx, start_idx + text_len, input_ids_i.device())?
                                    .unsqueeze(0)?
                                    .broadcast_as((3usize, text_len as usize))?;
                            llm_pos_ids_list.push(pos_ids);
                            let range_tensor = Tensor::arange(
                                0,
                                llm_grid_t,
                                input_ids_i.device(),
                            )?
                            .unsqueeze(D::Minus1)?;
                            let expanded_range = range_tensor
                                .broadcast_as((llm_grid_t as usize, (llm_grid_h * llm_grid_w) as usize))?;
                            let time_tensor = expanded_range.broadcast_mul(&Tensor::new(
                                vec![second_per_grid_t * self.cfg.vision_config.tokens_per_second as u32],
                                input_ids_i.device(),
                            )?)?.broadcast_add(&Tensor::new(
                                vec![start_idx + text_len],
                                input_ids_i.device(),
                            )?)?;
                            let t_index = time_tensor.flatten_all()?;
                            let h_index = Tensor::arange(
                                start_idx + text_len,
                                start_idx + text_len + llm_grid_h,
                                input_ids_i.device(),
                            )?
                            .unsqueeze(0)?
                            .unsqueeze(D::Minus1)?
                            .broadcast_as((
                                llm_grid_t as usize,
                                llm_grid_h as usize,
                                llm_grid_w as usize,
                            ))?
                            .flatten_all()?;
                            let w_index = Tensor::arange(
                                start_idx + text_len,
                                start_idx + text_len + llm_grid_w,
                                input_ids_i.device(),
                            )?
                            .unsqueeze(0)?
                            .unsqueeze(0)?
                            .broadcast_as((
                                llm_grid_t as usize,
                                llm_grid_h as usize,
                                llm_grid_w as usize,
                            ))?
                            .flatten_all()?;

                            let thw_index = Tensor::stack(&[t_index, h_index, w_index], 0)?;
                            llm_pos_ids_list.push(thw_index);
                            text_start = text_end + llm_grid_t * llm_grid_h * llm_grid_w;
                        }
                    }
                    Err(e) => {
                        println!("get vision_indices err: {}", e);
                    }
                };

                if text_start < input_ids_i.dim(0)? as u32 {
                    let start_idx = if llm_pos_ids_list.len() > 0 {
                        llm_pos_ids_list[llm_pos_ids_list.len() - 1]
                            .max_all()?
                            .to_scalar::<u32>()?
                            + 1
                    } else {
                        0
                    };
                    let text_len = input_ids_i.dim(0)? as u32 - text_start;
                    let pos_ids =
                        Tensor::arange(start_idx, start_idx + text_len, input_ids_i.device())?
                            .unsqueeze(0)?
                            .broadcast_as((3usize, text_len as usize))?;
                    llm_pos_ids_list.push(pos_ids);
                }
                let llm_position = Tensor::cat(&llm_pos_ids_list, 1)?.reshape((3, 1, ()))?;
                position_ids = position_ids
                    .slice_assign(&[(0..3), (i..i + 1), (0..input_ids.dim(1)?)], &llm_position)?;
                let position_deltas = llm_position.max_all()?.to_scalar::<u32>()? as i64 + 1
                    - input_ids_i.dim(0)? as i64;
                mrope_position_deltas.push(position_deltas);
            }

            let mut mrope_position_deltas = Tensor::new(mrope_position_deltas, input_ids.device())?;
            if mrope_position_deltas.rank() == 1 {
                mrope_position_deltas = mrope_position_deltas.unsqueeze(0)?;
            }
            return Ok((position_ids.contiguous()?, mrope_position_deltas));
        } else {
            if mask.is_some() {
                let mut position_ids = mask
                    .unwrap()
                    .to_dtype(candle_core::DType::F64)?
                    .cumsum(D::Minus1)?
                    .to_dtype(candle_core::DType::U32)?
                    .broadcast_sub(&Tensor::new(vec![1_u32], input_ids.device())?)?;
                for i in 0..position_ids.dim(0)? {
                    let mut position_ids_i = position_ids.i(i)?;
                    let mask_i = mask.unwrap().i(i)?;
                    // 如果有pad, 将填充位置置为1
                    // 当bs>1, 可能存在不同序列长度，需要添加pad使seq_len长度一致
                    if mask_i.sum_all()?.to_scalar::<u32>()? != mask_i.dim(0)? as u32 {
                        let zero_indices = zero_index(&mask_i)?;
                        let replace_1 = Tensor::ones(
                            zero_indices.dim(0)?,
                            candle_core::DType::U32,
                            input_ids.device(),
                        )?;
                        position_ids_i = position_ids_i
                            .scatter(&zero_indices, &replace_1, 0)?
                            .unsqueeze(0)?;
                        position_ids = position_ids.slice_assign(
                            &[(i..i + 1), (0..position_ids.dim(1)?)],
                            &position_ids_i,
                        )?;
                    }
                }
                position_ids = position_ids
                    .unsqueeze(0)?
                    .broadcast_as((3, input_ids.dim(0)?, input_ids.dim(1)?))?
                    .contiguous()?;
                let mut mrope_position_deltas = position_ids
                    .max(0)?
                    .max(D::Minus1)?
                    .broadcast_sub(&Tensor::new(
                        vec![mask.unwrap().dim(D::Minus1)? as u32 - 1],
                        input_ids.device(),
                    )?)?
                    .contiguous()?;
                if mrope_position_deltas.rank() == 1 {
                    mrope_position_deltas = mrope_position_deltas.unsqueeze(0)?;
                }
                return Ok((position_ids, mrope_position_deltas));
            } else {
                let position_ids =
                    Tensor::arange(0_u32, input_ids.dim(D::Minus1)? as u32, input_ids.device())?
                        .unsqueeze(0)?
                        .unsqueeze(0)?
                        .broadcast_as((3, input_ids.dim(0)?, input_ids.dim(D::Minus1)?))?
                        .contiguous()?;
                let mrope_position_deltas = Tensor::zeros(
                    (input_ids.dim(0)?, 1),
                    input_ids.dtype(),
                    input_ids.device(),
                )?;
                Ok((position_ids, mrope_position_deltas))
            }
        }
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        image_grid_thw: Option<&Tensor>,
        pixel_values_video: Option<&Tensor>,
        video_grid_thw: Option<&Tensor>,
        mask: &Tensor,
        cache_position: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        // input_ids shape: (bs, seq_len)
        let mut inputs_embeds = self.model.embed_tokens.forward(&input_ids)?;
        // inputs_embeds shape: (bs, seq_len, hidden_dim)
        if pixel_values.is_some() && image_grid_thw.is_some() {
            // image_embed shape: (seq_len, hidden_dim)
            let image_embed = self
                .visual
                .forward(pixel_values.unwrap(), image_grid_thw.unwrap())?;
            let vision_mask = get_equal_mask(&input_ids, self.cfg.image_token_id as u32)?;

            let n_image_tokens = vision_mask.sum_all()?.to_scalar::<u32>()?;
            if n_image_tokens as usize != image_embed.dim(0)? {
                return Err(Error::Msg(format!(
                    "n_image_token num: {} not equal to image_embed len: {}",
                    n_image_tokens,
                    image_embed.dim(0)?
                )));
            }
            inputs_embeds = masked_scatter_dim0(&inputs_embeds, &image_embed, &vision_mask)?;
        }
        if pixel_values_video.is_some() && video_grid_thw.is_some() {
            let video_embed = self
                .visual
                .forward(pixel_values_video.unwrap(), video_grid_thw.unwrap())?;

            let vision_mask = get_equal_mask(&input_ids, self.cfg.video_token_id as u32)?;
            let n_video_tokens = vision_mask.sum_all()?.to_scalar::<u32>()?;
            if n_video_tokens as usize != video_embed.dim(0)? {
                return Err(Error::Msg(format!(
                    "n_image_token num: {} not equal to image_embed len: {}",
                    n_video_tokens,
                    video_embed.dim(0)?
                )));
            }
            inputs_embeds = masked_scatter_dim0(&inputs_embeds, &video_embed, &vision_mask)?;
        }
        let mut position_ids;
        let mut rope_deltas;
        if (cache_position.is_some() && cache_position.unwrap().i(0)?.to_scalar::<u32>()? == 0)
            || self.rope_deltas.is_none()
        {
            (position_ids, rope_deltas) = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                Some(mask),
            )?;
            self.rope_deltas = Some(rope_deltas);
        } else {
            let (bs, seq_len, _) = inputs_embeds.dims3()?;
            let delta = if cache_position.is_some() {
                cache_position
                    .unwrap()
                    .i(0)?
                    .to_dtype(self.rope_deltas.as_ref().unwrap().dtype())?
                    .broadcast_add(&self.rope_deltas.as_ref().unwrap())?
                    .contiguous()?
                    .to_dtype(candle_core::DType::U32)?
            } else {
                Tensor::zeros(1, inputs_embeds.dtype(), inputs_embeds.device())?
            };
            position_ids = Tensor::arange(0u32, seq_len as u32, input_ids.device())?
                .unsqueeze(0)?
                .broadcast_as((bs, seq_len))?
                .broadcast_add(&delta)?
                .unsqueeze(0)?
                .broadcast_as((3, bs, seq_len))?
                .contiguous()?;
        }
        let outputs = self.model.forward(
            &inputs_embeds,
            seqlen_offset,
            Some(&position_ids)
        )?;
        let seq_len = outputs.dim(1)?;
        // let hidden_state = outputs.narrow(1, seq_len - 1, 1)?;
        let hidden_state = outputs;
        let logits = self.lm_head.forward(&hidden_state)?;
        let logits = logits.narrow(1, seq_len - 1, 1)?;
        Ok(logits)
    }
    pub fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }
}
