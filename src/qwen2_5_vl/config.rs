use candle_nn::Activation;
pub enum ModelType {
    Qwen2_5VLVision,
    Qwen2_5VLText,
}

pub enum BaseConfigKey {
    VisionConfig,
    TextConfig,
}

#[allow(unused)]
pub struct Qwen2_5VLVisionConfig {
    pub model_type: ModelType,
    pub base_config_key: BaseConfigKey,
    pub depth: usize,
    pub hidden_size: usize,
    pub hidden_act: Activation,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub spatial_patch_size: usize,
    pub temporal_patch_size: usize,
    pub tokens_per_second: usize,
    pub window_size: usize,
    pub out_hidden_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub fullatt_block_indexes: Vec<usize>,
}

impl Qwen2_5VLVisionConfig {
    pub fn init(
        depth: usize,
        hidden_size: usize,
        hidden_act: Activation,
        intermediate_size: usize,
        num_heads: usize,
        in_channels: usize,
        patch_size: usize,
        spatial_merge_size: usize,
        spatial_patch_size: usize,
        temporal_patch_size: usize,
        tokens_per_second: usize,
        window_size: usize,
        out_hidden_size: usize,
        rms_norm_eps: f64,
        rope_theta: f32,
        fullatt_block_indexes: Vec<usize>,
    ) -> Self {
        Self {
            model_type: ModelType::Qwen2_5VLVision,
            base_config_key: BaseConfigKey::VisionConfig,
            depth,
            hidden_size,
            hidden_act,
            intermediate_size,
            num_heads,
            in_channels,
            patch_size,
            spatial_merge_size,
            spatial_patch_size,
            temporal_patch_size,
            tokens_per_second,
            window_size,
            out_hidden_size,
            rms_norm_eps,
            rope_theta,
            fullatt_block_indexes,
        }
    }
    pub fn qwen2_5_vl_3_b_vision_config() -> Self {
        let depth = 32;
        let hidden_size = 1280;
        let hidden_act = Activation::Silu;
        let intermediate_size = 3420;
        let num_heads = 16;
        let in_channels = 3;
        let patch_size = 14;
        let spatial_merge_size = 2;
        let spatial_patch_size = 14;
        let temporal_patch_size = 2;
        let tokens_per_second = 2;
        let window_size = 112;
        let out_hidden_size = 2048;
        let rms_norm_eps = 1e-6;
        let rope_theta = 10000.0;
        let fullatt_block_indexes = vec![7, 15, 23, 31];

        Qwen2_5VLVisionConfig::init(
            depth,
            hidden_size,
            hidden_act,
            intermediate_size,
            num_heads,
            in_channels,
            patch_size,
            spatial_merge_size,
            spatial_patch_size,
            temporal_patch_size,
            tokens_per_second,
            window_size,
            out_hidden_size,
            rms_norm_eps,
            rope_theta,
            fullatt_block_indexes,
        )
    }
}

#[derive(Debug, Clone)]
pub enum RopeType {
    MRope,
}

#[allow(unused)]
#[derive(Debug, Clone)]
pub struct RopeScaling {
    pub rope_type: RopeType,
    pub mrope_section: Vec<usize>,
}

impl RopeScaling {
    pub fn qwen2_5_vl_3_b_rope() -> Self {
        Self {
            rope_type: RopeType::MRope,
            mrope_section: vec![16, 24, 24],
        }
    }
}

#[allow(unused)]
pub struct Qwen2_5VLTextConfig {
    pub model_type: ModelType,
    pub base_config_key: BaseConfigKey,
    pub vocab_size: usize,
    pub attention_dropout: f32,
    pub hidden_size: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub initializer_range: f32,
    pub rms_norm_eps: f64,
    pub use_cache: bool,
    pub tie_word_embeddings: bool,
    pub rope_theta: f32,
    pub use_sliding_window: bool,
    pub sliding_window: usize,
    pub max_window_layers: usize,
    pub rope_scaling: RopeScaling,
}

impl Qwen2_5VLTextConfig {
    pub fn init(
        vocab_size: usize,
        attention_dropout: f32,
        hidden_size: usize,
        intermediate_size: usize,
        num_hidden_layers: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        hidden_act: Activation,
        max_position_embeddings: usize,
        initializer_range: f32,
        rms_norm_eps: f64,
        use_cache: bool,
        tie_word_embeddings: bool,
        rope_theta: f32,
        use_sliding_window: bool,
        sliding_window: usize,
        max_window_layers: usize,
        rope_scaling: RopeScaling,
    ) -> Self {
        let head_dim = hidden_size / num_attention_heads;
        Self {
            model_type: ModelType::Qwen2_5VLText,
            base_config_key: BaseConfigKey::TextConfig,
            vocab_size,
            attention_dropout,
            hidden_size,
            head_dim,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            hidden_act,
            max_position_embeddings,
            initializer_range,
            rms_norm_eps,
            use_cache,
            tie_word_embeddings,
            rope_theta,
            use_sliding_window,
            sliding_window,
            max_window_layers,
            rope_scaling,
        }
    }

    pub fn qwen2_5_vl_3_b_text_config() -> Self {
        let vocab_size = 151936;
        let attention_dropout = 0.0;
        let hidden_size = 2048;
        let intermediate_size = 11008;
        let num_hidden_layers = 36;
        // let num_hidden_layers = 1;
        let num_attention_heads = 16;
        let num_key_value_heads = 2;
        let hidden_act = Activation::Silu;
        let max_position_embeddings = 128000;
        let initializer_range = 0.02;
        let rms_norm_eps = 1e-06;
        let use_cache = true;
        let tie_word_embeddings = true;
        let rope_theta = 1000000.0;
        let use_sliding_window = false;
        let sliding_window = 32768;
        let max_window_layers = 70;
        let rope_scaling = RopeScaling::qwen2_5_vl_3_b_rope();
        Qwen2_5VLTextConfig::init(
            vocab_size,
            attention_dropout,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            hidden_act,
            max_position_embeddings,
            initializer_range,
            rms_norm_eps,
            use_cache,
            tie_word_embeddings,
            rope_theta,
            use_sliding_window,
            sliding_window,
            max_window_layers,
            rope_scaling,
        )
    }
}

#[allow(unused)]
pub struct Qwen2_5VLConfig {
    pub vision_config: Qwen2_5VLVisionConfig,
    pub text_config: Qwen2_5VLTextConfig,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
    pub vision_start_token_id: usize,
    pub vision_end_token_id: usize,
    pub vision_token_id: usize,
    pub image_token_id: usize,
    pub video_token_id: usize,
}

impl Qwen2_5VLConfig {
    pub fn init(
        vision_config: Qwen2_5VLVisionConfig,
        text_config: Qwen2_5VLTextConfig,
        bos_token_id: usize,
        eos_token_id: usize,
        vision_start_token_id: usize,
        vision_end_token_id: usize,
        vision_token_id: usize,
        image_token_id: usize,
        video_token_id: usize,
    ) -> Self {
        Self {
            vision_config,
            text_config,
            bos_token_id,
            eos_token_id,
            vision_start_token_id,
            vision_end_token_id,
            vision_token_id,
            image_token_id,
            video_token_id,
        }
    }

    pub fn qwen2_5_vl_3_b_config() -> Self {
        let vision_config = Qwen2_5VLVisionConfig::qwen2_5_vl_3_b_vision_config();
        let text_config = Qwen2_5VLTextConfig::qwen2_5_vl_3_b_text_config();
        let bos_token_id = 151643;
        let eos_token_id = 151645;
        let vision_start_token_id = 151652;
        let vision_end_token_id = 151653;
        let vision_token_id = 151654;
        let image_token_id = 151655;
        let video_token_id = 151656;
        Qwen2_5VLConfig::init(
            vision_config,
            text_config,
            bos_token_id,
            eos_token_id,
            vision_start_token_id,
            vision_end_token_id,
            vision_token_id,
            image_token_id,
            video_token_id,
        )
    }
}

pub struct VisionSetting {
    pub image_factor: u32,
    pub min_pixels: u32,
    pub max_pixels: u32,
    pub max_ratio: u32,
    pub temporal_patch_size: usize,
    pub patch_size: usize,
    pub merge_size: usize,
    pub video_min_pixels: u32,
    pub video_max_pixels: u32,
    pub video_total_pixels: u32,
    pub frame_factor: u32,
    pub fps: f32,
    pub fps_min_frames: u32,
    pub fps_max_frames: u32,
    pub image_mean: Vec<f32>,
    pub image_std: Vec<f32>,
}

impl VisionSetting {
    pub fn default() -> Self {
        Self {
            image_factor: 28,
            min_pixels: 4 * 28 * 28,
            max_pixels: 16384 * 28 * 28,
            // max_pixels: 1000 * 28 * 28,
            max_ratio: 200,
            temporal_patch_size: 2,
            patch_size: 14,
            merge_size: 2,
            video_min_pixels: 128 * 28 * 28,
            video_max_pixels: 768 * 28 * 28,
            video_total_pixels: 24576 * 28 * 28,
            frame_factor: 2,
            fps: 2.0,
            fps_min_frames: 4,
            fps_max_frames: 768,
            image_mean: vec![0.48145466_f32, 0.4578275, 0.40821073],
            image_std: vec![0.26862954, 0.26130258, 0.27577711]
        }
    }
}
