import numpy as np
import torch
import pickle
import os
import pandas as pd
import spacy
from collections import Counter
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import transforms

# 确保先下载 Spacy 模型
# 执行: python -m spacy download en_core_web_sm
spacy_eng = spacy.load("en_core_web_sm")
# spacy_eng = spacy.load("/home/shaoxuan_xu/Temp/test/mmml-proj/img2caption/en_core_web_sm-3.8.0-py3-none-any.whl")

class Vocabulary:
    def __init__(self, freq_threshold):
        # 初始化词汇表，包括特殊标记
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentences):
        idx = 4  # 从索引4开始，因为0-3已经被占用
        frequency = Counter()

        for sentence in sentences:
            for word in self.tokenizer_eng(sentence):
                frequency[word] += 1
                if frequency[word] == self.freq_threshold:
                    self.itos[idx] = word
                    self.stoi[word] = idx
                    idx += 1

    def numericalize(self, sentence):
        tokenized_text = self.tokenizer_eng(sentence)
        return [self.stoi.get(word, self.stoi["<UNK>"]) for word in tokenized_text]


class FlickrDataset(Dataset):
    def __init__(self, 
                 root_dir="/root/autodl-tmp/flickr8k/Images", 
                 caption_path="/root/autodl-tmp/flickr8k/captions.txt", 
                 freq_threshold=5, 
                 transform=None, 
                 selecting_samples=10000, 
                 max_length=50,
                 adapter=None):
        """
        Args:
            root_dir (string): 图片所在的目录
            caption_path (string): 包含 captions 的文件路径
            freq_threshold (int): 词汇表中词汇的最低频率
            transform (callable, optional): 可选的转换函数
            selecting_samples (int or 'all'): 选择的样本数量
            max_length (int): captions 的固定长度
        """
        self.freq_threshold = freq_threshold
        self.transform = transform
        self.root_dir = root_dir
        self.max_length = max_length
        self.adapter = adapter
        # 读取 captions 文件
        if selecting_samples == 'all':
            self.df = pd.read_csv(caption_path)
        else:
            self.df = pd.read_csv(caption_path).iloc[:selecting_samples]

        self.captions = self.df['caption']
        self.images = self.df['image']

        # 构建词汇表
        self.vocab = Vocabulary(freq_threshold)
        self.captions_all = pd.read_csv("/root/autodl-tmp/flickr8k/all_captions.csv")['caption']
        self.vocab.build_vocabulary(self.captions_all.tolist())
        self.saw = True
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        image = self.images[index]
        img_path = os.path.join(self.root_dir, image)
        img = Image.open(img_path).convert("RGB")

        if self.adapter == 'mlp':
            img = self.transform(img)
        elif self.adapter == 'qformer':
            img = self.transform["eval"](img)

        # 数值化 caption
        # if self.saw:
        #     print(caption)
        #     self.saw = False
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        # 截断或填充到固定长度
        if len(numericalized_caption) < self.max_length:
            numericalized_caption += [self.vocab.stoi["<PAD>"]] * (self.max_length - len(numericalized_caption))
        else:
            numericalized_caption = numericalized_caption[:self.max_length]

        return img, torch.tensor(numericalized_caption)

def split_train_val_dataset(caption_path="/root/autodl-tmp/flickr8k/captions.txt",
                            train_ratio=0.8):
    caption_df = pd.read_csv(caption_path, delimiter=',', header=None, names=['image', 'caption'])
    img_caption_dict = {}
    
    for i in range(len(caption_df)):
        img = caption_df.iloc[i]['image']
        caption = caption_df.iloc[i]['caption']
        if img not in img_caption_dict:
            img_caption_dict[img] = []
        img_caption_dict[img].append(caption)
    
    # 对img_caption_dict进行划分
    img_list = list(img_caption_dict.keys())
    train_img_list = img_list[:int(len(img_list)*train_ratio)]
    val_img_list = img_list[int(len(img_list)*train_ratio):]
    
    train_caption_df = pd.DataFrame(columns=['image', 'caption'])
    val_caption_df = pd.DataFrame(columns=['image', 'caption'])
    
    # 选择每张图片的第一个caption
    for img in train_img_list:
        first_caption = img_caption_dict[img][0]  # 选择第一个caption
        train_caption_df = train_caption_df._append({'image': img, 'caption': first_caption}, ignore_index=True)
    
    for img in val_img_list:
        first_caption = img_caption_dict[img][0]  # 选择第一个caption
        val_caption_df = val_caption_df._append({'image': img, 'caption': first_caption}, ignore_index=True)
    
    all_caption_df = pd.concat([train_caption_df, val_caption_df], ignore_index=True)
    train_caption_df.to_csv("/root/autodl-tmp/flickr8k/train_captions.csv", index=False)
    val_caption_df.to_csv("/root/autodl-tmp/flickr8k/val_captions.csv", index=False)
    all_caption_df.to_csv("/root/autodl-tmp/flickr8k/all_captions.csv", index=False)

class FlickrDatasetQFormer(Dataset):
    def __init__(self, 
                 root_dir="/root/autodl-tmp/flickr8k/Images", 
                 caption_path="/root/autodl-tmp/flickr8k/captions.txt",
                 transform=None):
        """
        Args:
            root_dir (string): 图片所在的目录
            caption_path (string): 包含 captions 的文件路径
            freq_threshold (int): 词汇表中词汇的最低频率
            transform (callable, optional): 可选的转换函数
            selecting_samples (int or 'all'): 选择的样本数量
            max_length (int): captions 的固定长度
        """
        self.root_dir = root_dir
        self.df = pd.read_csv(caption_path)
        self.captions = self.df['caption']
        self.images = self.df['image']
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        image_name = self.images[index]
        img_path = os.path.join(self.root_dir, image_name)
        img = Image.open(img_path).convert("RGB")

        img = self.transform["eval"](img)
        return image_name, img, caption