import torch
import argparse
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from src.models import CaptionModel
from src.dataset import FlickrDataset, split_train_val_dataset, FlickrDatasetQFormer
from src.utils import save_checkpoint, setup_seed, get_loader
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from lavis.models import load_model_and_preprocess
os.environ["TOKENIZERS_PARALLELISM"] = "false"

####################################################################
# Functions
####################################################################

def train_and_eval(args, model, 
                  train_loader, val_loader,
                  optimizer, scheduler, filename, vocab):
    """
    训练并评估模型。

    参数：
    - args: 包含训练参数的命名空间，例如 num_epochs 和 device。
    - model: 要训练的模型。
    - train_loader: 训练数据的 DataLoader。
    - val_loader: 验证数据的 DataLoader。
    - optimizer: 优化器。
    - scheduler: 学习率调度器。
    - filename: 用于保存模型检查点的文件名。
    - vocab: 词汇表对象，包含词到索引的映射（vocab.stoi）。
    """

    # 定义损失函数，忽略填充项的损失
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
    best_val_loss = float('inf')
    BLEU = False
    for epoch in range(args.num_epochs):
        ########################
        #        训练阶段       #
        ########################
        model.train()
        train_loss = 0.0
        train_total = 0.0

        train_loader_tqdm = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{args.num_epochs}", total=len(train_loader))
        for imgs, captions in train_loader_tqdm:
            imgs = imgs.to(args.device)
            captions = captions.to(args.device)
            optimizer.zero_grad()
            outputs = model(imgs, captions[:, :-1])  # 输入为字幕的前一部分
            targets = captions[:, 1:]

            outputs = outputs.reshape(-1, outputs.size(2))  # 形状：[batch_size * (seq_len-1), vocab_size]
            targets = targets.reshape(-1)                   # 形状：[batch_size * (seq_len-1)]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * targets.size(0)
            train_total += targets.size(0)
            train_loader_tqdm.set_postfix(loss=train_loss / train_total, lr=optimizer.param_groups[0]['lr'])

        # 计算平均训练损失
        avg_train_loss = train_loss / train_total

        ########################
        #        验证阶段       #
        ########################
        model.eval()  # 设置模型为评估模式
        val_loss = 0.0
        val_total = 0.0
        
        val_loader_tqdm = tqdm(val_loader, desc="Validation", total=len(val_loader))
        for imgs, captions in val_loader_tqdm:
            imgs = imgs.to(args.device)          # 将图像数据移动到设备
            captions = captions.to(args.device)  # 将字幕数据移动到设备
            outputs = model(imgs, captions[:, :-1])
            targets = captions[:, 1:]
            
            outputs = outputs.reshape(-1, outputs.size(2))
            targets = targets.reshape(-1)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * targets.size(0)
            val_total += targets.size(0)
            val_loader_tqdm.set_postfix(loss=val_loss / val_total)

        # 计算平均验证损失
        avg_val_loss = val_loss / val_total

        # 打印本轮训练和验证的损失
        print(f"Epoch [{epoch+1}/{args.num_epochs}] | Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")
        if BLEU:
            smoothing_fn = SmoothingFunction().method1
            references, hypotheses = [], []
            count = 0
            with torch.no_grad():
                eval_loader = tqdm(val_loader, desc="Evaluation", total=len(val_loader))
                for imgs, captions in eval_loader:
                    imgs = imgs.to(args.device)
                    captions = captions.to(args.device)

                    generated_captions = model.captionBatch(imgs, vocab)
                    for i in range(len(imgs)):
                        ref = captions[i].tolist()
                        if count == 0:
                            print(ref)
                            print([[vocab.itos[idx] for idx in ref if idx not in {vocab.stoi["<SOS>"], vocab.stoi["<EOS>"], vocab.stoi["<PAD>"]}]])
                            print(generated_captions[i])
                            count+=1
                        references.append([[vocab.itos[idx] for idx in ref if idx not in {vocab.stoi["<SOS>"], vocab.stoi["<EOS>"], vocab.stoi["<PAD>"]}]])
                        hypotheses.append(generated_captions[i])

            # 计算 BLEU 分数
            print(references[0], hypotheses[0])
            avg_bleu = corpus_bleu(references, hypotheses, smoothing_function=smoothing_fn)
            print(f"Validation Set Average BLEU Score: {avg_bleu:.4f}")
        # 如果当前验证损失优于最佳验证损失，则保存模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            save_checkpoint(checkpoint, filename)
            print(f"Validation loss decreased. Model saved to {filename}")

        # 调整学习率
        scheduler.step()

def train_and_eval_qformer(args, model, train_loader, val_loader, 
                           optimizer, scheduler, filename):
    """
    训练并评估模型。

    参数：
    - args: 包含训练参数的命名空间，例如 num_epochs 和 device。
    - model: 要训练的模型。
    - train_loader: 训练数据的 DataLoader。
    - val_loader: 验证数据的 DataLoader。
    - optimizer: 优化器。
    - scheduler: 学习率调度器。
    - filename: 用于保存模型检查点的文件名。
    """

    # 定义损失函数，忽略填充项的损失
    best_val_loss = float('inf')
    for epoch in range(args.num_epochs):
        ########################
        #        训练阶段       #
        ########################
        model.train()
        train_loss = 0.0
        train_total = 0.0

        train_loader_tqdm = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{args.num_epochs}", total=len(train_loader))
        for _, imgs, captions in train_loader_tqdm:
            imgs = imgs.to(args.device)
            captions = captions
            optimizer.zero_grad()
            loss = model(imgs, captions)
    
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_total += 1
            train_loader_tqdm.set_postfix(loss=train_loss / train_total, lr=optimizer.param_groups[0]['lr'])

        # 计算平均训练损失
        avg_train_loss = train_loss / train_total

        ########################
        #        验证阶段       #
        ########################
        model.eval()  # 设置模型为评估模式
        val_loss = 0.0
        val_total = 0.0
        
        val_loader_tqdm = tqdm(val_loader, desc="Validation", total=len(val_loader))
        for _, imgs, captions in val_loader_tqdm:
            imgs = imgs.to(args.device)
            loss = model(imgs, captions)
            val_loss += loss.item()
            val_total += 1
            val_loader_tqdm.set_postfix(loss=val_loss / val_total)

        # 计算平均验证损失
        avg_val_loss = val_loss / val_total

        # 打印本轮训练和验证的损失
        print(f"Epoch [{epoch+1}/{args.num_epochs}] | Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")

        # 如果当前验证损失优于最佳验证损失，则保存模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            save_checkpoint(checkpoint, filename)
            print(f"Validation loss decreased. Model saved to {filename}")

        # 调整学习率
        scheduler.step()

if __name__ == '__main__':

    ####################################################################
    # Arguments
    ####################################################################

    parser = argparse.ArgumentParser(description='Flickr8k')
    parser.add_argument('-f', default='', type=str)

    parser.add_argument('--embed_size', type=int, default=512, metavar='N', help='embedding size (default: 256)')
    parser.add_argument('--batch_size', type=int, default=24, metavar='N', help='batch size (default: 24)')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate (default: 2e-3)')
    parser.add_argument('--optim', type=str, default='Adam', help='optimizer to use (default: Adam)')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs (default: 50)')
    parser.add_argument('--when_decay', type=int, default=30, help='when to decay learning rate (default: 30)')
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument('--num_layers', type=int, default=6, help='number of Transformer decoder layers (default: 6)')
    parser.add_argument('--nhead', type=int, default=8, help='number of heads in the Transformer decoder (default: 8)')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='dimension of the feedforward network in Transformer (default: 2048)')
    parser.add_argument('--adapter', type=str, default='qformer', help='the adapter used for image-text alignment')
    
    args = parser.parse_args()
    setup_seed(args.seed)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ####################################################################
    # Load the dataset 
    ####################################################################
    if args.adapter == 'qformer':
        print('Loading QFormer feature extractor')
        _, transform, _ = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=args.device)
    elif args.adapter == 'mlp':
        transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225]) 
                ]
            )

    # 加载训练和验证集
    path = "/root/autodl-tmp/flickr8k"
    if not os.path.exists(path + "/train_captions.csv") or not os.path.exists(path + "/val_captions.csv") or not os.path.exists(path + "/all_captions.csv"):
        print('Spliting dataset')
        split_train_val_dataset()

    train_loader, train_set = get_loader(caption_path=path + "/train_captions.csv", 
                                         transform=transform, 
                                         freq_threshold=2, 
                                         selecting_samples='all', 
                                         batch_size=args.batch_size, 
                                         adapter=args.adapter)
    val_loader, val_set = get_loader(caption_path=path + "/val_captions.csv", 
                                     transform=transform, 
                                     freq_threshold=2, 
                                     selecting_samples='all', 
                                     batch_size=args.batch_size, 
                                     shuffle= False, 
                                     adapter=args.adapter)
    loader, dataset = get_loader(caption_path=path + "/all_captions.csv", 
                                 transform=transform, 
                                 freq_threshold=2, 
                                 selecting_samples='all', 
                                 batch_size=args.batch_size, 
                                 adapter=args.adapter)

    args.vocab_size = len(dataset.vocab) if args.adapter=='mlp' else '32000 (TinyLlama)'
    # print(dataset.vocab.stoi)
    print("VocabSize:", args.vocab_size)
    print("train_set:", len(train_set))
    print("val_set:", len(val_set))
    

    ####################################################################
    # Training 
    ####################################################################

    model = CaptionModel(
        embed_size=args.embed_size,
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        dropout=True,
        encoder_type='CLIP',
        adapter = args.adapter
    ).to(device=args.device)

    if model.adapter == 'qformer':
        print(f'Using Encoder: ViT-L/14, Decoder: TinyLlama, Adapter: {model.adapter}, Using Prompt: {model.prompt}')
        print(f'Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
        print(f'Number of total parameters: {sum(p.numel() for p in model.parameters())}')
    elif model.adapter == 'mlp':
        print(f'Using Encoder: {model.encoder_type}, Decoder: TransformerDecoder, Adapter: {model.adapter}')
    
    if args.optim.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.when_decay, 0.1)
    filename = "caption_model_lr{}_decay{}_bsize{}.pth.tar".format(args.lr, args.when_decay, args.batch_size)

    if args.adapter == 'mlp':
        print('Training with MLP Adapter for Image-text Alignment')
        train_and_eval(args, model, train_loader, val_loader, optimizer, scheduler, filename, dataset.vocab)
    if args.adapter == 'qformer':
        print('Training with QFormer Adapter for Image-text Alignment')
        train_and_eval_qformer(args, model, train_loader, val_loader, optimizer, scheduler, filename)

