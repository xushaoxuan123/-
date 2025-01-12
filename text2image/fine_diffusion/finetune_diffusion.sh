export MODEL_NAME="/home/jiahao_zhao/多模态机器学习/期末大作业/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="/home/jiahao_zhao/多模态机器学习/期末大作业/diffusion-lora-128rank-16bs"
export DATASET_NAME="/home/jiahao_zhao/多模态机器学习/期末大作业/data/archive/train_images"
export CUDA_VISIBLE_DEVICES=1

accelerate launch --mixed_precision="bf16"  train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=16 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=1000 \
  --learning_rate=1e-4 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --checkpointing_steps=100 \
  --validation_prompt="A naruto with blue eyes." \
  --seed=1337 \
  --rank=128