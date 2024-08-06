# Clip Text to Image

使用 CLIP 模型完成“文搜图”和“图搜图”。离线的个人图片库。


## 项目初衷

几乎人人都有自己的图片库。
所以存放在哪里、使用是否方便可能会是问题。

1. 大多人乐意使用云盘，大概就是斯丹达尔所谓的“幸福的大多数”；
2. 有些人不是很喜欢但能接受上传到陌生云盘，至少用着的时候很方便；
3. 有些人无法接受这种安排，就是免费也不行，宣称使用多么方便也不行；
4. 有些人压根就没想过这回事。

我恰是那第三种人，或许也没有那么不幸，至少建造点东西还是可以收获一些快乐的。哪怕有点简陋。


## 项目情况

- 推理部分本地运行毫无压力，速度一般, 搜索的文图匹配表现尚可；

- 微调模型
    + GPU时代：得借助服务器（24G rtxA5000, 约1小时）；
    + GPU穷鬼：一度很担心把机器给跑烧了，但还是手痒没停下：入门消费级别显卡（6G）跑了一夜跑46%左右。

- 当自己感觉搜索出来的结果不是那么令人满意的时候，考虑两个地方：
    + FAISS 的搜索引擎：FAISS 向量检索引擎有多种基于向量相似度检索实现方式，兼顾向量库规
        模、搜索精度以及搜索速度（据称十万级别图像库直接用 `IndexFlat` 索引类型无压力）。
    + CN-CLIP 多模态模型：在 CLIP 的基础上用中文数据集继续训练而得。本项目在推理时所依赖
        的是 CN-CLIP 在 Flickr30k-CN 数据集上进行微调得到的模型。因此，如果你的“个人”图
        像库图像内容与这些数据集图像内容差异较大，则需要基于自己的图像库进行微调以保证一
        定的准确度。

- 增加个人图片
    + 个人图片库的基本存储形式变成FAISS的Index类型，从而利用FAISS完成相似度检索。
    + 当前是将原始图片存放在 `data/raw/weixin_img/`，将生成的 FAISS 存储文件放在
      `data/processed/xxx.index`
    + 实际使用中，多半是外接存储盘中存储个人图片库，那么本质上只需要生成对应的 FAISS 存
      储文件即可。但原始图片的命名（如果有）具有重要信息，所以可以考虑额外存储相匹配图片
      文件名。
    + 如果个人图片库规模较大（百万级别），则应该考虑以训练时用的base64数据类型存放和处理
    + 将图片及其命名转换成base64的字节数据的过程，可以参考
      `src/data/utils/transform_raw_images.py`

- TODO: 继续DIY
    + 再加个树莓派和家庭NAS，其实真就可以说是爱咋咋地。
    + 后续看看怎么加入语音搜索能力。
        所谓“多模态”，没有语音其实不算，而一旦涉及音频数据，那个计算量，想象一个“仅”10秒
        钟的音频（44.1kHz采样频率、16位编码，约2M）或视频（1080p分辨率、30fps、每帧32位
        颜色深度，约240M）。再训练一个本身参数量已经巨大的模型... 没有开源，没有可能。

- 两个文搜图示例：

<div style="display: flex; justify-content: space-between;">
  <img src="./data/external/images/fmh_OldBoys_blur.png" alt="Image 1" style="width: 45%;"/>
  <img src="./data/external/images/fmh_textQueryImage_02.png" alt="Image 2" style="width: 45%;"/>
</div>


## 完整项目目录

模型的微调直接参考[Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP)说明文档，其中
微调时可能会有 torch 版本过高导致的一些问题，我是直接安装 `pip install torch==2.1` 解决。

- `queryApp` 基本只涉及模型推理，所以是假设所有数据已经准备好，否则需要先生成相关数据。

- `src/data/utils/transform_raw_images.py` 用于将个人图片转换为能够被CLIP模型所接受的数
  据格式，从而继续微调CLIP模型。（Chinese-CLIP 官网给出了较为具体的指导，这个是个人的一
  个实现，仅供参考。）同时也可以作为规模数据处理的一个备用数据格式。

```sh
├── Chinese-CLIP    # 基础模型
│   ├── assets
│   ├── cn_clip
│   │   ├── clip
│   │   ├── deploy
│   │   ├── eval
│   │   ├── preprocess
│   │   └── training
│   ├── examples
│   ├── models
│   │   └── hfLLMs
│   ├── notebooks
│   ├── run_scripts
│   └── src
│       └── data    # transform_raw_images.py
├── fmhData         # 数据部分
│   ├── datasets
│   │   ├── Flickr30k-CN    # 共享数据用于模型微调
│   │   └── XiaoMuZai       # 个人数据
│   └── experiments         # CN-CLIP 模型微调过程数据
│       └── flickr30kcn_finetune_vit-l-14_roberta-base_batchsize64_1gpu
└── queryApp        # 应用部分
    ├── data
    │   ├── external
    │   ├── interim
    │   ├── processed
    │   └── raw
    ├── notebooks
    └── src
        ├── data    # utils/transform_raw_images.py
        ├── model
        └── visualization
```
