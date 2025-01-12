# 使用Latent Diffusion Model实现文生图
## 环境配置

建议使用environment.yml文件配置环境，安装命令如下：
```bash
    conda env create -f environment.yml
    conda activate ldm
```
## 数据集

只需将Flickr8k中的Image文件夹拖放到`latent_diffusion_model/data/flickr8k`目录下即可。

## 代码运行
进入`latent_diffusion_model/tools`目录。先训练VQ-VAE模型获得潜空间表示(训练参数见`latent_diffusion_model/config/flickr8k.yaml`)：
```bash
python train_vqvae.py
```
训完之后会保存VQ-VAE模型到`flickr8k/vqvae_autoencoder_ckpt.pth`。然后训练文生图的Latent Diffusion模型(训练参数见`latent_diffusion_model/config/flickr8k_text_cond.yaml`)：
```bash
python train_ddpm__cond.py
```
训完之后会保存Diffusion模型到`ddpm_ckpt_text_cond_clip.pth`。最后使用训练好的模型生成图像：
```bash
python sample_ddpm_text_cond.py
```

