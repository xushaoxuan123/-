import torch
from torch.utils.data import DataLoader
from src.dataset import FlickrDataset, FlickrDatasetQFormer
from PIL import Image
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test_model_bleu(model,val_loader,args, vocab):
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
    # print(references[0], hypotheses[0])
    avg_bleu = corpus_bleu(references, hypotheses, smoothing_function=smoothing_fn)
    print(f"Validation Set Average BLEU Score: {avg_bleu:.4f}")

def load_captions(captions_file='/root/autodl-tmp/flickr8k/captions.txt'):
    """
    从 captions 文件中加载所有图片的 captions，并返回一个字典。
    
    Args:
        captions_file (str): captions.txt 文件的路径。
    
    Returns:
        dict: 键为 img_name，值为该图片的所有参考 captions 的列表。
    """
    captions_dict = {}
    with open(captions_file, 'r') as f:
        next(f)  # 跳过标题行
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_id, caption = line.split(',', 1)  # 仅在第一个逗号处分割
            img_name = img_id.strip()
            caption = caption.strip()
            if img_name not in captions_dict:
                captions_dict[img_name] = []
            captions_dict[img_name].append(caption)
    return captions_dict

def test_model_bleu_llama(model, val_loader, args):
    smoothing_fn = SmoothingFunction().method1
    references, hypotheses = [], []
    captions_dict = load_captions()
    results_for_saving = []

    with torch.no_grad():
        eval_loader = tqdm(val_loader, desc="Evaluation", total=len(val_loader))
        for img_names, imgs, captions in eval_loader:
            imgs = imgs.to(args.device)
            generated_captions = model.captionBatch(images=imgs, vocabulary=None, max_length=30, prompt=args.prompt)
            for i in range(len(imgs)):
                img_name = img_names[i]
                generated_caption = generated_captions[i]
                hypotheses.append(generated_caption)
                if img_name in captions_dict:
                    ref_caps = captions_dict[img_name]
                    references.append([ref_cap.split() for ref_cap in ref_caps])
                else:
                    print(f"警告: 未找到图片 {img_name} 的参考 captions")
                    raise(NotImplementedError)
                
                # 保存结果到列表
                results_for_saving.append(f"{img_name},{generated_caption}\n")

    tokenized_hypotheses = [hyp.split() for hyp in hypotheses]
    bleu1 = corpus_bleu(references, tokenized_hypotheses, smoothing_function=smoothing_fn, weights=(1,0,0,0))
    bleu2 = corpus_bleu(references, tokenized_hypotheses, smoothing_function=smoothing_fn, weights=(0.5,0.5,0,0))
    bleu3 = corpus_bleu(references, tokenized_hypotheses, smoothing_function=smoothing_fn, weights=(0.333,0.333,0.334,0))
    bleu4 = corpus_bleu(references, tokenized_hypotheses, smoothing_function=smoothing_fn, weights=(0.25, 0.25, 0.25, 0.25))
    print(f"BLEU-1:{bleu1:.4f},BLEU-2:{bleu2:.4f},BLEU-3:{bleu3:.4f},BLEU-4:{bleu4:.4f}")
    
    # 保存生成的 img_names 和 captions 到一个 txt 文件
    output_file = 'generated_captions.txt'
    with open(output_file, 'w') as f:
        f.writelines(results_for_saving)
    print(f"captions saved to {output_file}")
    
    return bleu1, bleu2, bleu3, bleu4

class Collate:
    def __init__(self, pad_value):
        self.pad_value = pad_value
    
    def __call__(self, batch):
        imgs = torch.stack([item[0] for item in batch], dim=0)  # [batch_size, C, H, W]
        captions = torch.stack([item[1] for item in batch], dim=1).permute(1, 0)  # [batch_size, max_length]
        return imgs, captions

def get_loader(root_dir="/root/autodl-tmp/flickr8k/Images", 
               caption_path="/root/autodl-tmp/flickr8k/captions.txt", 
               transform=None, 
               batch_size=48, 
               freq_threshold = 5, 
               num_workers=8, 
               shuffle=True, 
               pin_memory=True, 
               selecting_samples = 10000,
               adapter=None):
    if adapter == 'mlp':
        dataset = FlickrDataset(root_dir=root_dir, 
                                caption_path=caption_path, 
                                transform=transform, 
                                freq_threshold=freq_threshold, 
                                selecting_samples=selecting_samples, 
                                adapter=adapter)
        pad_value = dataset.vocab.stoi["<PAD>"]
        loader = DataLoader(dataset=dataset, 
                            batch_size=batch_size, 
                            num_workers=num_workers, 
                            shuffle=shuffle, 
                            pin_memory=pin_memory,
                            collate_fn=Collate(pad_value), 
                            generator=torch.Generator(device='cpu'))
        return loader, dataset
    elif adapter == 'qformer':
        print("Loading dataset and dataloader for QFormer")
        dataset = FlickrDatasetQFormer(root_dir=root_dir,
                                       caption_path=caption_path,
                                       transform=transform)
        loader = DataLoader(dataset=dataset, 
                            batch_size=batch_size, 
                            num_workers=num_workers, 
                            shuffle=shuffle, 
                            pin_memory=pin_memory)
        return loader, dataset

def save_checkpoint(state, filename):
    print("saving checkpoint!")
    torch.save(state, filename)

def load_checkpoint(name, model):
    checkpoint = torch.load(name, map_location=model.device if hasattr(model, 'device') else 'cpu', weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])