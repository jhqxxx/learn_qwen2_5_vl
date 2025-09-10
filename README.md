### model config
### model key
* visual
    * patch_embed.proj.weight: Tensor[dims 1280, 3, 2, 14, 14; bf16, cuda:0]
    * blocks
        * attn
            * qkv
                * bias
                * weight
            * proj
                * weight
                * bias
        * mlp
            * down_proj
                * weight
                * bias
            * gate_proj
                * weight
                * bias
            * up_proj
                * weight
                * bias
        * norm1.weight
        * norm2.weight
    * merger
        * mlp
            * 0
                * bias
                * weight
            * 2
                * bias
                * weight
        * ln_q.weight


* model
    * embed_tokens.weight
    * layers
        * input_layernorm.weight
        * self_attn
            * q_proj
                * weight
                * bias
            * k_proj
                * weight
                * bias
            * v_proj
                * weight
                * bias
            * o_proj.weight
        * post_attention_layernorm.weight
        * mlp
            * up_proj.weight
            * gate_proj.weight
            * down_proj.weight
    * norm.weight

### model struct
#### text model
* embedding
* decoder layers
* RMSNorm
* rotary embedding

#### vision model
* vision patch embed
    * conv3d
* vision rotary embedding
* vision blocks
    * vision attention
        * window attention???
* patch merger


## data processor
* 输入：
```json
{
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "file:////mnt/d/messy/qwen-vl-test.png",
                },
                {
                    "type": "video",
                    "video": "file:////mnt/d/file/about_rust/cut_video/chapter_end.mp4",
                },
                {"type": "text", "text": "请分析图片和视频，提取其中所有可见文本内容，按从左到右、从上到下的布局，返回纯文本"},
            ],
        }
    ]
}
```

1. messages数组apply chat template，得到text数据：

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|video_pad|><|vision_end|>请分析图片和视频，提取其中所有可见文本内容，按从左到右、从上到下的布局，返回纯文本<|im_end|>
<|im_start|>assistant
```

2. 预处理type为image和video
* 遍历messages，将content里面有类型为image/image_url/video的提取出来
* 处理image
    * 根据路径前缀，如"http://", "https://","file://","data:image"，打开图片
    * smart_resize: 将图片大小智能resize成最接近28的倍数

* 处理video
    * 读取视频
    * 获取视频总帧数，帧率信息，
    * 智能计算每秒的帧数
    * 根据帧数提取图片

* 返回
    * image: 返回PILimage类型数组, mode=RGB
    * video：返回shape为(t, c, h, w)的tensor

3. 处理text，images，videos数据
* images
    * / 255，将数据处理到0-1之间
    * normalize
    OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
    * images shape: (img_num, c, h, w)
    * 分别处理每张图片：
        * 若img_num % temporal_patch_size ！= 0，repeat最后一个img以满足img_num % temporal_patch_size = 0
        * grid_t = img_num / temporal_patch_size
        * grid_h = h / patch_size
        * grid_w = w / patch_size
        * image reshape ->: (
                grid_t,
                temporal_patch_size,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
        * image transpose(0, 3, 6, 4, 7, 2, 1, 5, 8) -> (
                grid_t, 
                grid_h // merge_size, 
                grid_w // merge_size, 
                merge_size, 
                merge_size, 
                channel, 
                temporal_patch_size, 
                patch_size, 
                patch_size,
            )
        * image reshape -> (grid_t×grid_h×grid_w, channel×temporal_patch_size×patch_size×patch_size)
        * image grid -> (grid_t, grid_h, grid_w)
    * 组合多张图片：
        * 多张图片如果尺寸不一样，那么grid_t,grid_h，grid_w可能不一样，但是因为最后一维的维度是一致的，所以可以将它们在0维位置上拼接在一起,使得可以同时处理不同尺寸的图像
        * images shape -> (grid_t1×grid_h1×grid_w1+grid_t2×grid_h2×grid_w2+..., channel×temporal_patch_size×patch_size×patch_size)
        * images grid shape -> (img_num, 3)

* videos
    * 重新采样，可选
    * 按hw给视频分组
    * 分别处理每组视频
        * / 255
        * normalize
        * 视频比图像多了一个batchsize的维度，其他的和上面的差不多

* text
    * 原来text中一个<|image_pad|>代表这里是一个图像填充
    * 遍历text中所有<|image_pad|>，每个位置使用对应图像的真实token个数填充<|image_pad|>
        * 使用对应图像的grid shape：grid_t×grid_h×grid_w / merge_size 得到图像token个数：num_token
        * 该位置填充num_token个<|image_pad|>
    * 原来text中一个<|video_pad|>代表这里是一个视频填充
    * 遍历text中所有<|video_pad|>，每个位置使用对应视频的真实token个数填充<|video_pad|>，与图像类似
    * 将text经过tokenize enbed得到词嵌入

* 返回 text embed, image tensor, image grid tensor, video tensor, video grid tensor

## Generation
* 计算rope需要的position_ids
    * 输入(bs, seqlen)
    * 如果输入有图像或视频网格信息
        * position_ids(3, bs, seqlen)
        * 如果bs>1,不同seq可能有填充，需要传入attention_mask确定填充token位置
        * 如果attention_mask为None，生成一个维度与输入一致的全一tensor
        * 遍历每个batch
            * 根据attention_mask选出有效token
            * 找出seq中所有<|vision_start|>对应的token id所在的索引 vision_start_indexs
            * 根据vision_start_indexs + 1位置的token_id得到是添加的image的数量和video的数量
            * 循环image_num+video_num次
                * 确定下一个要处理的视觉pad是image还是video
                * 如果是image
                    * 根据对应image_id拿到该图像的网格信息
                    * 得到image_id前的text_len
                    * (0..text_len) + start_index生成text position序列再expand(3, text_len),text的position t,h,w的数据都一样
                    * 按该图像的grid生成position_id(3, image_token_len)
                    * 第一行是时间维度，第二行是h，第三行是w,
                * 如果是video，按照video数据处理，和image内容差不多
            * 将最后的text_token处理完
        * 合并所有的position
    * 如果没有图像或视频信息，直接处理text，它得到的position_ids中t,h,w数据都一样


* 依赖ffmpeg安装参考: https://github.com/zmwangx/rust-ffmpeg/wiki/Notes-on-building
    * ubuntu/WSL: sudo apt install -y clang libavcodec-dev libavformat-dev libavutil-dev pkg-config libavfilter-dev libavdevice-dev