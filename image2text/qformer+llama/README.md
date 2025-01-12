# Environment
The environment configuration of this repository is a bit of tricky, we recommend you to do the following steps:
```bash
    conda create -n mmml-proj python=3.11
    conda activate mmml-proj

    pip install salesforce-lavis
    pip install opencv-python
    pip install transformers==4.34
    pip install nltk
```
It seems that `transformers==4.34` conflicts with `salesforce-lavis==1.0.2` but in our experiments it turned out okay.

# Dataset

Flickr8K

# Method

Visual Encoder: ViT-L/14 (Inherited from QFormer)

Vision-Text Adapter: QFormer Block (To generate query tokens)

Text Decoder: TinyLlama/TinyLlama-1.1B-Chat-v1.0. For training and inference we follow the format 

- ```<Prompt><BOS><query token> ```

or without prompt

- ```<BOS><query token> ```

# Running script
For training model
```bash
HF_ENDPOINT=https://hf-mirror.com python main.py 
```
For testing model on the validation set
```
HF_ENDPOINT=https://hf-mirror.com python test.py 
```