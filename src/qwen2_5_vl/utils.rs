use base64::{Engine, engine::general_purpose};
use candle_core::{DType, Device, Error, IndexOp, Result, Tensor};
use ffmpeg_next as ffmpeg;
use image::{DynamicImage, ImageReader};
use num::integer::lcm;
use std::{
    fs::{self, File},
    io::{Cursor, Write},
};

use crate::config::VisionSetting;

pub fn nonzero_index_vec(mask: &Tensor) -> Result<Vec<u32>> {
    // 根据mask矩阵选出其中不为1的元素所在索引, 返回vec
    let mut mask = mask.clone();
    if mask.dtype() != DType::U32 {
        mask = mask.to_dtype(DType::U32)?;
    }
    match mask.rank() {
        0 => {
            return Err(Error::Msg(format!(
                "input rank must > 0, the input tensor rank: {}",
                mask.rank()
            )));
        }
        1 => {
            let mask_vector = mask.to_vec1::<u32>()?;
            let indices: Vec<u32> = mask_vector
                .iter()
                .enumerate()
                .filter_map(|(idx, &val)| if val != 0 { Some(idx as u32) } else { None })
                .collect();
            Ok(indices)
        }
        _ => {
            return Err(Error::Msg(format!(
                "input rank not support, the input tensor rank: {}",
                mask.rank()
            )));
        }
    }
}

pub fn nonzero_index(mask: &Tensor) -> Result<Tensor> {
    // 根据mask矩阵选出其中不为1的元素所在索引, 返回Tensor
    let index_vec = nonzero_index_vec(mask)?;
    let indices_tensor = Tensor::from_slice(&index_vec, index_vec.len(), mask.device())?;
    Ok(indices_tensor)
}

pub fn zero_index_vec(mask: &Tensor) -> Result<Vec<u32>> {
    // 根据mask矩阵选出其中不为1的元素所在索引, 返回vec
    let mut mask = mask.clone();
    if mask.dtype() != DType::U32 {
        mask = mask.to_dtype(DType::U32)?;
    }
    match mask.rank() {
        0 => {
            return Err(Error::Msg(format!(
                "input rank must > 0, the input tensor rank: {}",
                mask.rank()
            )));
        }
        1 => {
            let mask_vector = mask.to_vec1::<u32>()?;
            let indices: Vec<u32> = mask_vector
                .iter()
                .enumerate()
                .filter_map(|(idx, &val)| if val == 0 { Some(idx as u32) } else { None })
                .collect();
            Ok(indices)
        }
        _ => {
            return Err(Error::Msg(format!(
                "input rank not support, the input tensor rank: {}",
                mask.rank()
            )));
        }
    }
}

pub fn zero_index(mask: &Tensor) -> Result<Tensor> {
    let index_vec = zero_index_vec(mask)?;
    let indices_tensor = Tensor::from_slice(&index_vec, index_vec.len(), mask.device())?;
    Ok(indices_tensor)
}

pub fn nonzero_slice(mask: &Tensor) -> Result<Vec<(usize, usize)>> {
    // 根据mask矩阵选出其中不为1的元素所在索引
    // 根据索引获取连续索引间隔
    // 如不为零索引元素为[0, 3, 4, 5, 8, 9]
    // 间隔为: [(0, 1), (3, 6), (8, 10)]
    // 索引前闭后开, 所以end+1
    let mut index_vec = nonzero_index_vec(mask)?;
    match index_vec.len() {
        0 => {
            return Ok(vec![]);
        }
        1 => {
            return Ok(vec![(index_vec[0] as usize, (index_vec[0] + 1) as usize)]);
        }
        _ => {
            let mut vec_slice = vec![];
            let mut start = index_vec.remove(0);
            let mut last = start;

            for i in index_vec {
                if i == (last + 1) {
                    last = i;
                    continue;
                } else {
                    vec_slice.push((start as usize, (last + 1) as usize));
                    start = i;
                    last = i;
                }
            }
            vec_slice.push((start as usize, (last + 1) as usize));
            Ok(vec_slice)
        }
    }
}

pub fn masked_scatter_dim0(original: &Tensor, replace: &Tensor, mask: &Tensor) -> Result<Tensor> {
    // 根据mask中非0元素所在索引,使用replace中的数据替换掉original中的数据
    // original: rank = 3: (bs, seq_len, hidden_dim)
    // replace: rank = 2: (seq_len, hidden_dim)
    // mask: rank = 2: (bs, seq_len)
    // 推理时bs=1,为了方便替换,将bs squeeze,替换后再unsqueeze
    // 按行替换
    if original.dim(0)? != 1 || mask.dim(0)? != 1 {
        return Err(Error::Msg(format!(
            "masked_scatter_dim0 original bs: {} or mask bs :{} not equal to 1 ",
            original.dim(0)?,
            mask.dim(0)? != 1
        )));
    }
    let mut original = original.squeeze(0)?;
    let mask = mask.squeeze(0)?;
    let slices = nonzero_slice(&mask)?;
    let mut sub_start = 0usize;
    let mut sub_end = 0usize;
    for (start, end) in slices {
        sub_end = sub_start + (end - start);
        let sub_replace = replace.i((sub_start..sub_end, ..))?;
        original = original.slice_assign(&[(start..end), (0..original.dim(1)?)], &sub_replace)?;
        sub_start = sub_end;
    }
    original = original.unsqueeze(0)?;
    Ok(original)
}

pub fn get_equal_mask(input_ids: &Tensor, token_ids: u32) -> Result<Tensor> {
    let image_token_id_tensor = Tensor::new(vec![token_ids], input_ids.device())?;
    let mask = input_ids
        .broadcast_eq(&image_token_id_tensor)?
        .to_dtype(candle_core::DType::U32)?;
    Ok(mask)
}

pub fn get_vision_next_indices(input_ids: &Tensor, token_id: u32) -> Result<Tensor> {
    // input_ids -> shape: (seq_len)
    let mask = get_equal_mask(&input_ids, token_id)?;
    let indices = nonzero_index(&mask)?;
    let indices = indices.broadcast_add(&Tensor::new(vec![1u32], input_ids.device())?)?;
    Ok(indices)
}

pub fn get_template(path: String) -> Result<String> {
    let tokenizer_config_file = path.clone() + "/tokenizer_config.json";
    assert!(
        std::path::Path::new(&tokenizer_config_file).exists(),
        "tokenizer_config.json not exists in model path"
    );
    let tokenizer_config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(tokenizer_config_file)?)
            .map_err(|e| Error::Msg(format!("load tokenizer_config file error:{}", e)))?;
    let chat_template = tokenizer_config["chat_template"]
        .as_str()
        .ok_or(Error::Msg(format!("chat_template to str error")))?;
    // 修复模板中的问题行
    let fixed_template = chat_template
        .replace(
            "message.content.startswith('<tool_response>')",
            "message.content is startingwith('<tool_response>')", // 使用minijinja中的 is startingwith 替换
        )
        .replace(
            "message.content.endswith('</tool_response>')",
            "message.content is endingwith('</tool_response>')", // 使用minijinja中的 is endingwith 替换
        )
        .replace(
            "content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n')",
            "((content | split('</think>'))[0] | rstrip('\\n') | split('<think>'))[-1] | lstrip('\\n')", // 使用自定义的split, rstrip, lstrip过滤器替换
        )
        .replace(
            "content.split('</think>')[-1].lstrip('\\n')",
            "(content | split('</think>'))[-1] | lstrip('\\n')", // 使用自定义的过滤器替换
        )
        .replace(
            "reasoning_content.strip('\\n')",
            "reasoning_content | strip('\\n')", // 使用自定义的过滤器替换
        )
        .replace(
            "content.lstrip('\\n')",
            "content | lstrip('\\n')", // 使用自定义的过滤器替换
        );
    if fixed_template.contains(".split(") {
        println!(
            "-------------------------------- Warning: Template still contains .split() method calls"
        );
    }
    Ok(fixed_template)
}

pub fn find_safetensors_files(path: &str) -> Result<Vec<String>> {
    let mut files = Vec::new();

    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let file_path = entry.path();

        if file_path.is_file() {
            if let Some(extension) = file_path.extension() {
                if extension == "safetensors" {
                    files.push(file_path.to_string_lossy().to_string());
                }
            }
        }
    }

    Ok(files)
}

pub fn load_image_from_url(url: &str) -> Result<DynamicImage> {
    let response = reqwest::blocking::get(url)
        .map_err(|e| Error::Msg(format!("Failed to fetch image from url: {}", e)))?;
    let bytes = response
        .bytes()
        .map_err(|e| Error::Msg(format!("Failed to get image bytes: {}", e)))?;

    let cursor = Cursor::new(bytes);
    let img = ImageReader::new(cursor)
        .with_guessed_format()
        .map_err(|e| Error::Msg(format!("Failed to read image format: {}", e)))?
        .decode()
        .map_err(|e| Error::Msg(format!("Failed to decode image: {}", e)))?;
    Ok(img)
}

pub fn load_image_from_base64(base64_data: &str) -> Result<DynamicImage> {
    let image_data = general_purpose::STANDARD
        .decode(base64_data)
        .map_err(|e| Error::Msg(format!("Failed to decode image: {}", e)))?;
    let cursor = Cursor::new(image_data);
    let img = ImageReader::new(cursor)
        .with_guessed_format()
        .map_err(|e| Error::Msg(format!("Failed to read image format: {}", e)))?
        .decode()
        .map_err(|e| Error::Msg(format!("Failed to decode image: {}", e)))?;
    Ok(img)
}

pub fn round_by_factor(num: u32, factor: u32) -> u32 {
    let round = (num as f32 / factor as f32).round() as u32;
    round * factor
}

pub fn floor_by_factor(num: f32, factor: u32) -> u32 {
    let floor = (num / factor as f32).floor() as u32;
    floor * factor
}

pub fn ceil_by_factor(num: f32, factor: u32) -> u32 {
    let ceil = (num / factor as f32).ceil() as u32;
    ceil * factor
}

pub fn smart_resize(
    img_h: u32,
    img_w: u32,
    vision_setting: &VisionSetting,
    is_img: bool,
    video_ratio: Option<u32>,
) -> Result<(u32, u32)> {
    if std::cmp::max(img_h, img_w) / std::cmp::min(img_h, img_w) > vision_setting.max_ratio {
        return Err(Error::Msg(format!(
            "absolute aspect ratio mush be smaller than {}, got {}",
            vision_setting.max_ratio,
            std::cmp::max(img_h, img_w) / std::cmp::min(img_h, img_w)
        )));
    }
    let mut image_factor = vision_setting.image_factor;
    if let Some(ratio) = video_ratio {
        image_factor = lcm(image_factor, ratio);
    }
    let mut h_bar = std::cmp::max(image_factor, round_by_factor(img_h, image_factor));
    let mut w_bar = std::cmp::max(image_factor, round_by_factor(img_w, image_factor));
    let mut max_pixels = 0u32;
    let mut min_pixels = 0u32;
    if is_img {
        min_pixels = vision_setting.min_pixels;
        max_pixels = vision_setting.max_pixels;
    } else {
        min_pixels = vision_setting.video_min_pixels;
        max_pixels = vision_setting.video_max_pixels;
    }
    if h_bar * w_bar > max_pixels {
        let beta = ((img_h * img_w) as f32 / max_pixels as f32).sqrt();
        h_bar = floor_by_factor(img_h as f32 / beta, image_factor);
        w_bar = floor_by_factor(img_w as f32 / beta, image_factor);
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f32 / (img_h * img_w) as f32).sqrt();
        h_bar = ceil_by_factor(img_h as f32 * beta, image_factor);
        w_bar = ceil_by_factor(img_w as f32 * beta, image_factor);
    }
    Ok((h_bar, w_bar))
}

pub fn get_image(file: &String) -> Result<DynamicImage> {
    let mut img = None;
    if file.starts_with("http://") || file.starts_with("https://") {
        img = Some(load_image_from_url(&file)?);
    }
    if file.starts_with("file://") {
        let mut path = file.clone();
        path = path.split_off(7);
        img = Some(
            ImageReader::open(path)
                .map_err(|e| Error::Msg(format!("Failed to open file: {}", e)))?
                .decode()
                .map_err(|e| Error::Msg(format!("Failed to decode image: {}", e)))?,
        );
    }
    if file.starts_with("data:image") {
        if file.contains("base64,") {
            let data: Vec<&str> = file.split("base64,").collect();
            let data = data[1];
            img = Some(load_image_from_base64(data)?);
        }
    }
    if img.is_some() {
        return Ok(img.unwrap());
    }
    Err(Error::Msg("get image from message failed".to_string()))
}

fn save_file(
    frame: &ffmpeg::frame::Video,
    index: usize,
) -> std::result::Result<(), std::io::Error> {
    let mut file = File::create(format!("frame{}.ppm", index))?;
    file.write_all(format!("P6\n{} {}\n255\n", frame.width(), frame.height()).as_bytes())?;
    file.write_all(frame.data(0))?;
    Ok(())
}

pub fn get_video_data(
    file: &String,
    vision_setting: &VisionSetting,
    device: &Device,
) -> Result<Tensor> {
    ffmpeg::init().map_err(|e| Error::Msg(format!("Failed to initialize ffmpeg: {}", e)))?;

    let mut ictx = ffmpeg::format::input(&file)
        .map_err(|e| Error::Msg(format!("Failed to open video file: {}", e)))?;
    let input = ictx
        .streams()
        .best(ffmpeg::media::Type::Video)
        .ok_or_else(|| Error::Msg(format!("No video stream found")))?;
    let video_stream_index = input.index();
    let context_decoder = ffmpeg::codec::context::Context::from_parameters(input.parameters())
        .map_err(|e| Error::Msg(format!("Failed to crate decoder context: {}", e)))?;
    let mut decoder = context_decoder
        .decoder()
        .video()
        .map_err(|e| Error::Msg(format!("Failed to decoder video: {}", e)))?;

    let video_h = decoder.height();
    let video_w = decoder.width();
    let format = decoder.format();

    let frames = input.frames();
    let rate = (input.rate().0 as f32 / input.rate().1 as f32).round() as u32;
    // 1s取两帧
    let min_frames = ceil_by_factor(
        vision_setting.fps_min_frames as f32,
        vision_setting.frame_factor,
    );
    let max_frames = floor_by_factor(
        vision_setting.fps_max_frames as f32,
        vision_setting.frame_factor,
    );
    let nframes = (frames as f32 / rate as f32 * vision_setting.fps) as u32;
    let nframes = std::cmp::min(std::cmp::max(nframes, min_frames), max_frames);
    let nframes = round_by_factor(nframes, vision_setting.frame_factor);
    let sample_interval = (frames as f32 / nframes as f32).round() as u32;
    let mut frame_id = 0_u32;

    // 图片帧使用scaler reshape的时候需要保证宽高是16的倍数,不然reshape出来的是损坏的图片
    // 所以计算resize的目标宽高时,需要用16和image_factor的最小公倍数
    let (resize_h, resize_w) = smart_resize(video_h, video_w, vision_setting, false, Some(16))?;
    let mut scaler = ffmpeg::software::scaling::context::Context::get(
        format,
        video_w,
        video_h,
        ffmpeg::format::Pixel::RGB24,
        resize_w,
        resize_h,
        ffmpeg::software::scaling::flag::Flags::BILINEAR
            | ffmpeg::software::scaling::flag::Flags::ACCURATE_RND,
    )
    .map_err(|e| Error::Msg(format!("Failed to crate scaler: {}", e)))?;

    let mut frames_vec = Vec::new();
    let mut receive_and_process_decoded_frames =
        |decoder: &mut ffmpeg::decoder::Video| -> Result<()> {
            let mut decoded = ffmpeg::frame::Video::empty();
            while decoder.receive_frame(&mut decoded).is_ok() {
                if frame_id % sample_interval == 0 {
                    let mut rgb_frame = ffmpeg::frame::Video::empty();
                    scaler
                        .run(&decoded, &mut rgb_frame)
                        .map_err(|e| Error::Msg(format!("Failed to scaler run decoded: {}", e)))?;

                    // save_file(&rgb_frame, frame_id as usize);
                    let frame_data = rgb_frame.data(0);
                    let frame_tensor = Tensor::from_slice(
                        frame_data,
                        (resize_h as usize, resize_w as usize, 3),
                        device,
                    )?
                    .permute((2, 0, 1))?;
                    frames_vec.push(frame_tensor);
                }
                frame_id += 1;
            }
            Ok(())
        };

    for (stream, packet) in ictx.packets() {
        if stream.index() == video_stream_index {
            decoder
                .send_packet(&packet)
                .map_err(|e| Error::Msg(format!("Failed to send packet: {}", e)))?;
            receive_and_process_decoded_frames(&mut decoder)?;
        }
    }
    decoder
        .send_eof()
        .map_err(|e| Error::Msg(format!("Failed to decoder.send_eof(): {}", e)))?;
    receive_and_process_decoded_frames(&mut decoder)?;

    if frames_vec.is_empty() {
        return Err(Error::Msg("No frames extracted from video".to_string()));
    }
    // (t, c, h, w)
    let frames_tensor = Tensor::stack(&frames_vec, 0)?.contiguous()?;
    Ok(frames_tensor)
}

pub fn safe_arg_sort_last_dim(t: &Tensor, ascending: bool) -> Result<Tensor> {
    // tensor在GPU上时，维度超过1024， arg_sort_last_dim方法会报错
    // 所以维度大于1024时，放到CPU上处理
    let last_dim = t.dims()[t.rank() - 1];
    if last_dim <= 1024 {
        t.arg_sort_last_dim(ascending)
    } else {
        let cpu_tensor = t.to_device(&Device::Cpu)?;
        let sorted_indices = cpu_tensor.arg_sort_last_dim(ascending)?;
        sorted_indices.to_device(t.device())
    }
}
