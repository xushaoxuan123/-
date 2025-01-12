import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from diffusers import UNet2DConditionModel, DDPMScheduler
from dataclasses import dataclass
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import os
from tqdm import tqdm
from diffusers.optimization import get_cosine_schedule_with_warmup

@dataclass
class TrainingConfig:
    image_size = 128 
    train_batch_size = 32
    eval_batch_size = 2
    num_epochs = 5
    gradient_accumulation_steps = 1  # 增加梯度累积步数
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 1
    save_model_epochs = 1
    mixed_precision = "fp16"
    output_dir = "/home/jiahao_zhao/多模态机器学习/期末大作业/conditional-diffusion-model-base"
    # BERT配置
    max_text_length = 24
    # 训练配置
    seed = 42
    push_to_hub = False

class TextImageDataset(Dataset):
    def __init__(self, image_dir, caption_file, tokenizer, image_size=256, max_length=77):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 图像预处理
        self.image_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        # 加载CSV格式的caption文件
        import pandas as pd
        df = pd.read_csv(caption_file)
        
        # 假设CSV文件有file_name和text两列
        self.image_paths = [os.path.join(image_dir, fname) for fname in df['file_name']]
        self.captions = df['text'].tolist()
                
    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, idx):
        # 加载和预处理图像
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.image_transforms(image)
        
        # 处理文本
        caption = self.captions[idx]
        encoded = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "image": image,
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze()
        }

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
        # 冻结BERT参数
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        self.text_projection = nn.Sequential(
            nn.Linear(text_encoder_dim, cross_attention_dim),
            nn.GELU(),
            nn.LayerNorm(cross_attention_dim)
        )
        
    def forward(self, pixel_values, timesteps, input_ids, attention_mask):
        # 获取文本编码
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        text_embeds = text_outputs.last_hidden_state
        
        # 投影文本特征
        text_embeds = self.text_projection(text_embeds)
        
        # 条件UNet前向传播
        noise_pred = self.unet(
            pixel_values,
            timesteps,
            encoder_hidden_states=text_embeds,
        ).sample
        
        return noise_pred

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    global_step = 0
    
    for epoch in range(config.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        
        for step, batch in enumerate(train_dataloader):
            clean_images = batch["image"]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            # 采样噪声
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]
            # 采样随机时间步
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (bs,), device=clean_images.device
            )
            # 添加噪声
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            with accelerator.accumulate(model):
                # 预测噪声
                noise_pred = model(
                    noisy_images, 
                    timesteps, 
                    input_ids,
                    attention_mask
                )
                
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step
            }
            progress_bar.set_postfix(**logs)
            global_step += 1
            
        # 保存模型检查点
        if accelerator.is_main_process:
            if (epoch + 1) % config.save_model_epochs == 0:
                accelerator.save_state(f"{config.output_dir}/checkpoint-{epoch}")

def main():
    config = TrainingConfig()
    
    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained('/home/jiahao_zhao/多模态机器学习/期末大作业/BERT_Base')
    
    # 创建dataset
    dataset = TextImageDataset(
        image_dir="/home/jiahao_zhao/多模态机器学习/期末大作业/data/archive/train_images",
        caption_file="/home/jiahao_zhao/多模态机器学习/期末大作业/data/archive/train_images/metadata.csv",
        tokenizer=tokenizer,
        image_size=config.image_size,
        max_length=config.max_text_length
    )
    
    # 创建dataloader
    train_dataloader = DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True
    )
    
    # 初始化模型
    model = ConditionedUNet()
    
    # 创建noise scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # 学习率调度器
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs)
    )
    
    # 开始训练
    train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

if __name__ == "__main__":
    main()