use candle_core::{D, DType, Device, Error, IndexOp, Result, Tensor, safetensors};
use candle_nn::{VarBuilder, VarMap};
use candle_transformers::{generation::LogitsProcessor, models::deepseek2::SplitOp};
use qwen_2_5_vl::{
    config::{Qwen2_5VLConfig, Qwen2_5VLTextConfig, Qwen2_5VLVisionConfig},
    qwen2_5_vl::{
        processor::{GeneralInput, Qwen2_5VLProcessor},
        qwen2_5_vl_model::Qwen2_5_VLForConditionalGeneration,
        utils::{
            find_safetensors_files, get_template, nonzero_index, nonzero_slice,
            safe_arg_sort_last_dim,
        },
        vision_model::Qwen2_5VisionTransformerPretrainedModel,
    },
    rope::Qwen2_5VLTextRotaryEmbedding,
};
use std::{collections::HashSet, fs, thread, time, vec};
use tokenizers::Tokenizer;

pub fn generate_print(
    model: &mut Qwen2_5_VLForConditionalGeneration,
    logits_processor: &mut LogitsProcessor,
    processor: &Qwen2_5VLProcessor,
    message: &str,
    sample_len: usize,
    device: &Device,
) -> Result<()> {
    let mut input = processor.process_info(message)?;
    let mut seq_len = input.input_ids.dim(1)?;
    println!("seq_len: {:?}", seq_len);
    let mut seqlen_offset = 0;
    let eos_token = match processor
        .tokenizer
        .get_vocab(true)
        .get("<|endoftext|>")
        .copied()
    {
        Some(token) => token,
        None => return Err(Error::Msg("can't get <|endoftext|>".to_string())),
    };
    let eos_token2 = match processor
        .tokenizer
        .get_vocab(true)
        .get("<|im_end|>")
        .copied()
    {
        Some(token) => token,
        None => return Err(Error::Msg("can't get <|im_end|>".to_string())),
    };

    let mut input_ids = input.input_ids.clone();
    println!("input_ids: {:?}", input_ids);
    let mut pixel_values = if input.pixel_values.is_some() {
        Some(&input.pixel_values.unwrap().clone())
    } else {
        None
    };
    println!("pixel_values: {:?}", pixel_values);
    let image_grid_thw = if input.image_grid_thw.is_some() {
        Some(&input.image_grid_thw.unwrap().clone())
    } else {
        None
    };
    println!("image_grid_thw: {:?}", image_grid_thw);
    let mut pixel_values_video = if input.pixel_values_video.is_some() {
        Some(&input.pixel_values_video.unwrap().clone())
    } else {
        None
    };
    println!("pixel_values_video: {:?}", pixel_values_video);
    let video_grid_thw = if input.video_grid_thw.is_some() {
        Some(&input.video_grid_thw.unwrap().clone())
    } else {
        None
    };
    println!("video_grid_thw: {:?}", video_grid_thw);
    let mut mask = input.mask.clone();
    let mut cache_position = input.cache_position.clone();
    let second_per_grid_ts = if input.second_per_grid_ts.is_some() {
        Some(input.second_per_grid_ts.unwrap().clone())
    } else {
        None
    };

    let mut generate = Vec::new();
    for _ in 0..sample_len {
        let logits = model.forward(
            &input_ids,
            pixel_values,
            image_grid_thw,
            pixel_values_video,
            video_grid_thw,
            &mask,
            Some(&cache_position),
            seqlen_offset,
            second_per_grid_ts.clone(),
        )?;
        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;

        let next_token = logits_processor.sample(&logits)?;
        generate.push(next_token);

        if next_token == eos_token || next_token == eos_token2 {
            break;
        }
        seqlen_offset += seq_len;
        seq_len = 1;
        input_ids = Tensor::from_vec(vec![next_token], (1, 1), device)?;
        let appendd_mask = Tensor::ones((1, 1), mask.dtype(), device)?;
        mask = Tensor::cat(&[mask, appendd_mask], 1)?;
        cache_position = Tensor::from_vec(vec![seqlen_offset as u32], 1, device)?;
        pixel_values = None;
        pixel_values_video = None;
    }
    let res = processor.token_decode(generate)?;
    println!("generate: \n {:?}", res);
    model.clear_kv_cache();

    Ok(())
}

fn main() -> Result<()> {
    let device = Device::cuda_if_available(0)?;

    let model_path = "/mnt/c/jhq/huggingface_model/Qwen/Qwen2___5-VL-3B-Instruct";
    let template = get_template(model_path.to_string())?;
    let processor = Qwen2_5VLProcessor::new(model_path, &template, &device, DType::BF16)?;

    let message = r#"
    {
        "messages": [
            {
                "role": "user",
                "content": [ 
                    {
                        "type": "image",
                        "image": "file://./assets/ocr_test.png"
                    },               
                    {
                        "type": "text", 
                        "text": "请分析图片并提取所有可见文本内容，按从左到右、从上到下的布局，返回纯文本"
                    }
                ]
            }
        ]
    }
    "#;

    let model_list = find_safetensors_files(&model_path)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, DType::BF16, &device)? };
    let cfg = Qwen2_5VLConfig::qwen2_5_vl_3_b_config();
    let mut vl_generate = Qwen2_5_VLForConditionalGeneration::new(cfg, vb)?;
    let mut logits_processor = LogitsProcessor::new(5643, None, None);
    let _ = generate_print(
        &mut vl_generate,
        &mut logits_processor,
        &processor,
        message,
        512,
        &device,
    )?;

    Ok(())
}
