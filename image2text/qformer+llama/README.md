# 使用Q-former和TinyLlama实现图生文
## 环境配置

本仓库的环境配置有点棘手，我们建议您执行以下步骤：
```bash
    conda create -n qformer_llama python=3.11
    conda activate qformer_llama

    pip install salesforce-lavis
    pip install opencv-python
    pip install transformers==4.34
    pip install nltk
```
这里我们使用了`transformers==4.34`，安装时它似乎与`salesforce-lavis==1.0.2`冲突，但是在我们的实验中并没有发生问题。

## 数据集

我们只使用提供的Flickr8k 数据集进行实验。由于Flickr8k 数据集中一张图像对应5条文本，为避免模型混淆同一图像的多种表述，我们采用了单一标注策略，即每张图片仅对应一条文本，以减少模型对于同义不同表述文本的歧义识别。经过处理后，我们获得了8091个图像-文本对，并按照8:2的比例将其划分为训练集和验证集。

训练模型前，请将Images文件夹和caption.txt拖放到`data/flickr8k`目录下。

## 代码运行
训练模型：
```bash
HF_ENDPOINT=https://hf-mirror.com python main.py 
```
推理阶段：使用模型进行文生图
```bash
HF_ENDPOINT=https://hf-mirror.com python test.py 
```

# 方法

视觉编码器继承自Q-former的ViT-g/14，使用QFormer Block进行视觉文本模态的对齐。使用TinyLlama进行文本解码时，在训练阶段和推理阶段，我们遵循以下格式
```bash
<Prompt><BOS><query token> 
```
或
```bash
<Prompt><BOS><query token><SEP><answer token>
```
