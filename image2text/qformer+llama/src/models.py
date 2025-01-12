# model.py
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import math
import re
import clip
from lavis.models import load_model
from transformers import AutoModelForCausalLM, AutoTokenizer

class PositionalEncoding(nn.Module):
    """为Transformer添加位置信息的模块"""

    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, embed_size]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embed_size]
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

class ImageEncoder(nn.Module):
    def __init__(self, embed_size, freeze=True, dropout=True):
        super(ImageEncoder, self).__init__()
        self.freeze = freeze
        self.ResNet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # 只保留ResNet的卷积层部分
        self.ResNet = nn.Sequential(*list(self.ResNet.children())[:-2]) # output: [batch_size, 512, 7, 7]
        # 冻结ResNet参数
        if self.freeze:
            for param in self.ResNet.parameters():
                param.requires_grad = False    
        self.dropout = nn.Dropout(0.2)
        self.should_drop = dropout
        self.fc = nn.Linear(512, embed_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        features = self.ResNet(x)   # output: [batch_size, 512, 7, 7]
        features = features.view(features.size(0), features.size(1), -1)  # output: [batch_size, 512, 49]
        features = features.permute(0, 2, 1)  # output: [batch_size, 49, 512]
        features = self.fc(features)  # output: [batch_size, 49, embed_size]
        if self.should_drop:
            features = self.dropout(features)
        return features

class ImageEncoderCLIP(nn.Module):
    def __init__(self, embed_size, freeze=True, dropout=True, device='cpu'):
        super(ImageEncoderCLIP, self).__init__()
        self.freeze = freeze
        self.device = device
        # 加载预训练的 CLIP 模型
        self.clip_model, _ = clip.load("ViT-L/14", device=self.device)
        self.clip_model.eval()  # 设置为评估模式
        
        if self.freeze:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        self.dropout = nn.Dropout(0.2)
        self.should_drop = dropout
        # CLIP 的图像编码器输出维度，ViT-B/32 的 embed_size 是 512
        self.fc = nn.Linear(self.clip_model.visual.output_dim, embed_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, 3, H, W]
        Returns:
            features: Tensor, shape [batch_size, embed_size]
        """
        with torch.no_grad():
            features = self.clip_model.encode_image(x)  # [batch_size, clip_embed_size]
        if self.should_drop:
            features = self.dropout(self.fc(features))
        else:
            features = self.fc(features)
        return features

        
class TransformerDecoder(nn.Module):
    """使用Transformer解码器生成文本描述"""

    def __init__(self, embed_size, vocab_size, num_layers, nhead, dim_feedforward, dropout=0.1, encoder_type='ResNet'):
        super(TransformerDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        self.positional_encoding = PositionalEncoding(embed_size)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.embed_size = embed_size
        self.encoder_type = encoder_type
        
    def forward(self, captions, memory):
        """
        Args:
            captions: Tensor, shape [batch_size, seq_len]
            memory: Tensor, shape [batch_size, seq_len=49, embed_size]
        Returns:
            outputs: Tensor, shape [batch_size, seq_len, vocab_size]
        """
        embeddings = self.embed(captions) * math.sqrt(self.embed_size)  # [batch_size, seq_len, embed_size]
        embeddings = self.positional_encoding(embeddings)  # [batch_size, seq_len, embed_size]
        embeddings = embeddings.permute(1, 0, 2)  # [seq_len, batch_size, embed_size]

        if self.encoder_type == 'CLIP':
            # memory shape: [batch_size, embed_size]
            memory = memory.unsqueeze(0)    # [seq_len=1, batch_size, embed_size]
        else:
            # memory shape: [batch_size, 49, embed_size]
            memory = memory.permute(1, 0, 2)  # [seq_len=49, batch_size, embed_size]

        #print(f'captions: {captions}')
        tgt_mask, tgt_padding_mask = self.generate_square_subsequent_mask(captions.size(1), captions)
        #print(f'target mask: {tgt_mask}, {tgt_mask.size()}')
        #print(f'target padding mask: {tgt_padding_mask},{tgt_padding_mask.size()}')
        tgt_mask = tgt_mask.to(embeddings.device)
        tgt_padding_mask = tgt_padding_mask.to(embeddings.device)

        outputs = self.transformer_decoder(tgt=embeddings, 
                                           memory=memory, 
                                           tgt_mask=tgt_mask,
                                           tgt_key_padding_mask=tgt_padding_mask)
        outputs = outputs.permute(1, 0, 2)  # [batch_size, seq_len, embed_size]
        outputs = self.fc_out(outputs)  # [batch_size, seq_len, vocab_size]
        return outputs

    def generate_square_subsequent_mask(self, sz, input):
        tgt_mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
        tgt_padding_mask = (input == 0)
        tgt_padding_mask = tgt_padding_mask.to(dtype=torch.float32)
        return tgt_mask, tgt_padding_mask

class CaptionModel(nn.Module):
    """结合ImageEncoder和TransformerDecoder的完整模型"""
    def __init__(self, embed_size, vocab_size, num_layers, nhead, dim_feedforward, dropout=True, encoder_type='ResNet', adapter=None):
        super(CaptionModel, self).__init__()
        self.encoder_type = encoder_type
        self.adapter = adapter
        if adapter == 'mlp':
            if encoder_type == 'ResNet':
                self.encoder = ImageEncoder(embed_size=embed_size, freeze=True, dropout=True)
            elif encoder_type == 'CLIP':
                self.encoder = ImageEncoderCLIP(embed_size=embed_size, freeze=True, dropout=True)
            else:
                raise ValueError("Invalid encoder type. Please choose from ['ResNet', 'CLIP']")
            self.decoder = TransformerDecoder(embed_size, vocab_size, num_layers, nhead, dim_feedforward, dropout=0.1 if dropout else 0.0, encoder_type=self.encoder_type)
            self.embed_size = embed_size
            self.vocab_size = vocab_size
        elif adapter == 'qformer':
            self.qformer = load_model("blip2", "pretrain").float()
            self.tiny_llama = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            self.tiny_llama.eval()  # 确保 TinyLlama 进入评估模式
            for param in self.tiny_llama.parameters():
                param.requires_grad = False
            self.qformer_hidden_size = self.qformer.Qformer.config.hidden_size
            self.tinyllama_hidden_size = self.tiny_llama.config.hidden_size
            self.tinyllama_vocab_size = self.tiny_llama.vocab_size
            self.projector = nn.Linear(self.qformer_hidden_size, self.tiny_llama.config.hidden_size)
            self.tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            self.max_txt_len = 50
            self.prompt = True
    def forward(self, images, captions):
        """
        前向传播
        Args:
            images: Tensor, shape [batch_size, 3, H, W]
            captions: Tensor, shape [batch_size, seq_len]
        Returns:
            outputs: Tensor, shape [batch_size, seq_len, vocab_size]
        """
        if self.adapter == 'mlp':
            features = self.encoder(images)  # [batch_size, seq_len=49, embed_size]
            outputs = self.decoder(captions, features)  # [batch_size, seq_len, vocab_size]
            return outputs
        elif self.adapter == 'qformer':
            query_token, img_embed = self.qformer.forward_image(images)
            inputs_llama = self.projector(query_token)  # [batch_size, 32, llama_hidden_size]
            if self.prompt:
                prompt = "Generate a simple short caption:"
                prompt_tokens = self.tokenizer(
                    [prompt] * images.size(0),
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=9,  # 假设提示长度为10
                ).to('cuda')
                prompt_embeds = self.tiny_llama.get_input_embeddings()(prompt_tokens['input_ids'])  # [batch_size, prompt_length, llama_hidden_size]

            bos_token_id = self.tokenizer.bos_token_id
            bos_embed = self.tiny_llama.get_input_embeddings()(torch.tensor([[bos_token_id]]).to(images.device))   # [1, 1, llama_hidden_size]
            bos_embed = bos_embed.expand(inputs_llama.size(0), -1, -1)                             # [batch_size, 1, llama_hidden_size]
            inputs_llama = torch.cat([bos_embed, inputs_llama], dim=1)                             # [batch_size, 1+num_query_token, llama_hidden_size]
            if prompt:
                inputs_llama = torch.cat([prompt_embeds, inputs_llama], dim=1)                     # 用prompt的话，[batch_size, prompt_len+1+num_query_token, llama_hidden_size]
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(images.device)  # [batch_size, 1+num_query_token]
            
            caps = [c + "\n" for c in captions]
            caps_tokens = self.tokenizer(
                caps,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(images.device) # [batch_size, txt_len]
            
            targets = caps_tokens['input_ids'].masked_fill(caps_tokens['input_ids']==self.tokenizer.pad_token_id, -100) # [batch_size, txt_len]
            empty_targets = torch.ones((targets.size(0), atts_llama.size(1)), dtype=torch.long).to(images.device).fill_(-100)  # [batch_size, 1+num_query_token]
            targets = torch.cat([empty_targets, targets], dim=1)  # 标签：[batch_size, (prompt_len+)1+num_query_token+txt_len]
                                                                  # 前面的[batch_size, (prompt_len+)1+num_query_token]全都是-100, 模型看不到

            captions_embeds = self.tiny_llama.get_input_embeddings()(caps_tokens['input_ids'])  # [batch_size, txt_len, llama_hidden_size]
            
            input_embeds = torch.cat([inputs_llama, captions_embeds],dim=1)        # [batch_size, (prompt_len+)1+num_query_token+txt_len, llama_hidden_size]
            attention_mask = torch.cat([atts_llama, caps_tokens['attention_mask']], dim=1) # [batch_size, (prompt_len+)1+num_query_token+txt_len]
            outputs = self.tiny_llama(inputs_embeds=input_embeds, 
                                      attention_mask=attention_mask,
                                      labels=targets,
                                      return_dict=True)
            loss = outputs.loss                                 
            return loss         # cross entropy loss

    def captionBatch(self, images, vocabulary, max_length=50, prompt=True):
        """
        生成批量图片的描述
        Args:
            images: Tensor, shape [batch_size, 3, H, W]
            vocabulary: Vocabulary 对象
            max_length: 最大生成长度
        Returns:
            results: List[List[str]], shape [batch_size]
        """
        if self.adapter == 'mlp':
            device = images.device
            batch_size = images.size(0)
            results = [[] for _ in range(batch_size)]

            with torch.no_grad():
                features = self.encoder(images)  # [batch_size, seq_len=49, embed_size]

            # 初始化输入为 <SOS>
            inputs = torch.tensor([vocabulary.stoi["<SOS>"]] * batch_size, dtype=torch.long).to(device)  # [batch_size]
            inputs = inputs.unsqueeze(1)  # [batch_size, 1]

            for _ in range(max_length):
                outputs = self.decoder(inputs, features)  # [batch_size, seq_len, vocab_size]
                outputs = outputs[:, -1, :]  # [batch_size, vocab_size]
                _, predicted = outputs.max(1)  # [batch_size]

                predicted_words = [vocabulary.itos[p.item()] for p in predicted]
                for i, word in enumerate(predicted_words):
                    if word != "<EOS>":
                        results[i].append(word)
                inputs = torch.cat([inputs, predicted.unsqueeze(1)], dim=1)  # [batch_size, seq_len+1]

            return results
        elif self.adapter == 'qformer':
            self.qformer.eval()  # 设置模型为评估模式
            with torch.no_grad():
                query_token, img_embed = self.qformer.forward_image(images)  # [batch_size, 32, qformer_hidden_size]
                inputs_llama = self.projector(query_token)  # [batch_size, 32, llama_hidden_size]
                
                if prompt:
                    prompt = "Generate a simple short caption:"
                    prompt_tokens = self.tokenizer(
                        [prompt] * images.size(0),
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=9,  # 假设提示长度为10
                    ).to('cuda')
                    prompt_embeds = self.tiny_llama.get_input_embeddings()(prompt_tokens['input_ids'])  # [batch_size, prompt_length, llama_hidden_size]
                
                bos_token_id = self.tokenizer.bos_token_id
                bos_embed = self.tiny_llama.get_input_embeddings()(torch.tensor([[bos_token_id]]).to(images.device))   # [1, 1, llama_hidden_size]
                bos_embed = bos_embed.expand(inputs_llama.size(0), -1, -1)                             # [batch_size, 1, llama_hidden_size]
                inputs_llama = torch.cat([bos_embed, inputs_llama], dim=1)                             # [batch_size, 33, llama_hidden_size]
                if prompt:
                    inputs_llama = torch.cat([prompt_embeds, inputs_llama], dim=1)          # 用prompt的话，[batch_size, prompt_len+33, llama_hidden_size]
                atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(images.device)  # [batch_size, 33]

                generated_ids = self.tiny_llama.generate(
                    inputs_embeds=inputs_llama,                  # [batch_size, 33, llama_hidden_size]
                    attention_mask=atts_llama,                   # [batch_size, 33]
                    max_length=max_length,                       # 输入长度 + 生成的最大长度
                    num_beams=1,                                 # 束搜索的束宽度
                    # temperature=0.7,                             # 采样温度
                    # top_p=0.7,                                   # 核采样的累积概率
                    repetition_penalty=1.5,                      # 重复惩罚参数
                    eos_token_id=self.tokenizer.eos_token_id,    # 结束 Token
                    pad_token_id=self.tokenizer.pad_token_id,    # 填充 Token
                    do_sample=False,                             # 启用采样
                    # 可以根据需要添加其他参数
                )
                
                generated_captions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                
                processed_captions = []
                for caption in generated_captions:
                    match = re.match(r'^(.*?[\.\!\?])', caption)
                    if match:
                        first_sentence = match.group(1)
                    else:
                        first_sentence = caption
                    processed_captions.append(first_sentence)
            return processed_captions

    def captionImage(self, image, vocabulary, max_length=50):
        """
        生成单张图片的描述
        Args:
            image: Tensor, shape [3, H, W]
            vocabulary: Vocabulary 对象
            max_length: 最大生成长度
        Returns:
            result: List[str]
        """
        if self.adapter == 'mlp':
            device = image.device
            result = []
            with torch.no_grad():
                features = self.encoder(image)  # [1, seq_len=49, embed_size]

            # 初始化输入为 <SOS>
            inputs = torch.tensor([vocabulary.stoi["<SOS>"]], dtype=torch.long).to(device)  # [1]
            inputs = inputs.unsqueeze(0)  # [1, 1]

            for _ in range(max_length):
                outputs = self.decoder(inputs, features)  # [1, seq_len, vocab_size]
                outputs = outputs[:, -1, :]  # [1, vocab_size]
                _, predicted = outputs.max(1)  # [1]

                predicted_word = vocabulary.itos[predicted.item()]
                if predicted_word == "<EOS>":
                    break
                result.append(predicted_word)
                inputs = torch.cat([inputs, predicted.unsqueeze(0)], dim=1)  # [1, seq_len+1]

            return result
        