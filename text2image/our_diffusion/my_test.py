import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from diffusers import UNet2DConditionModel, DDPMScheduler
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
from dataclasses import dataclass
from safetensors.torch import load_file

# 复用训练代码中的配置和模型定义
@dataclass
class TrainingConfig:
    image_size = 128
    max_text_length = 24
    output_dir = "/home/jiahao_zhao/多模态机器学习/期末大作业/conditional-diffusion-model-base"

class ConditionedUNet(nn.Module):
    def __init__(self, text_encoder_dim=768, cross_attention_dim=512):
        super().__init__()
        self.unet = UNet2DConditionModel(
            sample_size=64,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 256),
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ),
            cross_attention_dim=cross_attention_dim,
        )
        
        self.text_encoder = BertModel.from_pretrained('/home/jiahao_zhao/多模态机器学习/期末大作业/BERT_Base')
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        self.text_projection = nn.Sequential(
            nn.Linear(text_encoder_dim, cross_attention_dim),
            nn.GELU(),
            nn.LayerNorm(cross_attention_dim)
        )
        
    def forward(self, pixel_values, timesteps, input_ids, attention_mask):
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        text_embeds = text_outputs.last_hidden_state
        text_embeds = self.text_projection(text_embeds)
        
        noise_pred = self.unet(
            pixel_values,
            timesteps,
            encoder_hidden_states=text_embeds,
        ).sample
        
        return noise_pred

def generate_image(text, model, tokenizer, noise_scheduler, device="cuda", num_inference_steps=50):
    """
    从文本生成图像
    Args:
        text (str): 输入的文本描述
        model (ConditionedUNet): 训练好的模型
        tokenizer: BERT tokenizer
        noise_scheduler: DDPMScheduler
        device (str): 使用的设备
        num_inference_steps (int): 推理步数
    Returns:
        PIL.Image: 生成的图像
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    # 将模型设置为评估模式
    model.eval()
    model = model.to(device)
    
    # 编码输入文本
    text_inputs = tokenizer(
        text,
        padding="max_length",
        max_length=24,  # 使用与训练时相同的长度
        truncation=True,
        return_tensors="pt"
    )
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    
    batch_size = 1
    noise = torch.randn(
        (batch_size, 3, 128, 128),
        device=device
    )
    
    # 设置推理步数
    noise_scheduler.set_timesteps(num_inference_steps)
    
    # 初始化为纯噪声
    latents = noise
    
    # 逐步去噪
    for t in tqdm(noise_scheduler.timesteps, desc="Generating image"):
        # 将时间步扩展到batch size
        timestep = torch.tensor([t] * batch_size, device=device)
        
        # 预测噪声残差
        with torch.no_grad():
            noise_pred = model(
                latents,
                timestep,
                text_inputs["input_ids"],
                text_inputs["attention_mask"]
            )
        
        # 执行去噪步骤
        latents = noise_scheduler.step(
            noise_pred,
            t,
            latents
        ).prev_sample
    
    # 缩放和解码为图像
    image = latents.squeeze().cpu().permute(1, 2, 0).numpy()
    image = ((image + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(image)
    
    return image

def main():
    # 初始化配置
    config = TrainingConfig()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 初始化tokenizer
    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('/home/jiahao_zhao/多模态机器学习/期末大作业/BERT_Base')
    
    # 初始化模型和scheduler
    print("Initializing model...")
    model = ConditionedUNet()
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    # 加载检查点
    checkpoint_path = os.path.join(config.output_dir, "checkpoint-4")
    print(f"Loading checkpoint from {checkpoint_path}")

    # 加载模型权重
    model_safetensors = os.path.join(checkpoint_path, "model.safetensors")
    if os.path.exists(model_safetensors):
        from safetensors.torch import load_file
        state_dict = load_file(model_safetensors)
    else:
        print("Falling back to model.bin")
        state_dict = torch.load(os.path.join(checkpoint_path, "model.bin"))
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    # 生成图像
    prompts = [
        "a wodden house"
    ]
    
    print("Generating images...")
    for i, text in enumerate(prompts):
        print(f"\nGenerating image for prompt: {text}")
        image = generate_image(
            text=text,
            model=model,
            tokenizer=tokenizer,
            noise_scheduler=noise_scheduler,
            device=device,
            num_inference_steps=100
        )
        
        # 保存图像
        output_path = f"generated_image_{i}.png"
        image.save(output_path)
        print(f"Saved image to {output_path}")

if __name__ == "__main__":
    main()