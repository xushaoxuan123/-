from diffusers import AutoPipelineForText2Image
import torch
import pandas as pd
import os
from collections import defaultdict
from tqdm import tqdm

picture_file = "/home/jiahao_zhao/多模态机器学习/期末大作业/test" # 我们选出的1091张在LoRA微调时没有直接使用的图片
name_list = os.listdir(picture_file)
caption_file = "/home/jiahao_zhao/多模态机器学习/期末大作业/data/archive/captions.txt"
name_dict = defaultdict(list)
with open(caption_file, "r") as f:
    for line in f:
        line = line.strip()
        tmp_name = line[:line.find(",")]
        if tmp_name in name_list:
            name_dict[tmp_name].append(line[line.find(",")+1:])

# for name in name_dict:
#     print(name_dict[name][0])

pipeline = AutoPipelineForText2Image.from_pretrained("/home/jiahao_zhao/多模态机器学习/期末大作业/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights("/home/jiahao_zhao/多模态机器学习/期末大作业/diffusion-lora-128rank-16bs/checkpoint-1000", weight_name="pytorch_lora_weights.safetensors")

for name in tqdm(name_list):
    if len(name_dict[name]) == 0:
        continue
    image = pipeline(name_dict[name][0], num_inference_steps=100).images[0]
    image.save(f"/home/jiahao_zhao/多模态机器学习/期末大作业/res_of_diffusion_ori/{name}")