import torch
import argparse
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from src.models import CaptionModel
from src.dataset import FlickrDataset, split_train_val_dataset, FlickrDatasetQFormer
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from lavis.models import load_model_and_preprocess
from src.utils import setup_seed, get_loader, load_checkpoint, test_model_bleu_llama

####################################################################
# Functions
####################################################################


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
        print('Loading QFormer feature extractor...')
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
    print('Successfully Loaded image transformation module!')
    # 加载训练和验证集
    path = "/root/autodl-tmp/flickr8k"
    if not os.path.exists(path + "/train_captions.csv") or not os.path.exists(path + "/val_captions.csv") or not os.path.exists(path + "/all_captions.csv"):
        print('Spliting dataset')
        split_train_val_dataset()

    loader, dataset = get_loader(caption_path=path + "/val_captions.csv", 
                                     transform=transform, 
                                     freq_threshold=2, 
                                     selecting_samples='all', 
                                     batch_size=args.batch_size, 
                                     shuffle=False, 
                                     adapter=args.adapter)

    args.vocab_size = len(dataset.vocab) if args.adapter=='mlp' else '32000 (TinyLlama)'
    args.prompt = True      # Use prompt

    print("VocabSize:", args.vocab_size)

    filename = "caption_model_lr{}_decay{}_bsize{}.pth.tar".format(args.lr, args.when_decay, args.batch_size)

    ####################################################################
    # Test on single images
    ####################################################################
    
    # 实例化新的模型用于测试
    test_model = CaptionModel(
        embed_size=args.embed_size,
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        dropout=True,
        encoder_type='CLIP',
        adapter = args.adapter
    ).to(device=args.device)
    
    print(f'Using Encoder: {test_model.encoder_type}, Decoder: TransformerDecoder, Adapter: {test_model.adapter}')

    load_checkpoint(filename, test_model)

    test_model_bleu_llama(test_model, loader, args)
