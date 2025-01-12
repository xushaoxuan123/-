from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.translate.rouge_score import rouge_n_sentence_level, rouge_l_sentence_level
from cider.cider import Cider
import numpy as np

def evaluate_metrics(hypotheses, references):
   """
   hypotheses: list of predicted captions
   references: list of ground truth captions (each can have multiple references)
   """
   # BLEU scores
   smooth = SmoothingFunction()
   bleu1 = corpus_bleu(references, hypotheses, weights=(1,0,0,0), 
                      smoothing_function=smooth.method1)
   bleu2 = corpus_bleu(references, hypotheses, weights=(0.5,0.5,0,0),
                      smoothing_function=smooth.method1) 
   bleu3 = corpus_bleu(references, hypotheses, weights=(0.33,0.33,0.33,0),
                      smoothing_function=smooth.method1)
   bleu4 = corpus_bleu(references, hypotheses, weights=(0.25,0.25,0.25,0.25),
                      smoothing_function=smooth.method1)
   
   # METEOR
   meteor = np.mean([meteor_score(refs, hyp) for hyp, refs in zip(hypotheses, references)])
   
   # ROUGE-L
   rouge_l = np.mean([rouge_l_sentence_level(hyp, refs) 
                     for hyp, refs in zip(hypotheses, references)])
   
   # CIDEr-D
   scorer = Cider()
   cider_score, _ = scorer.compute_score(references, hypotheses)
   
   return {
       'BLEU-1': bleu1*100,
       'BLEU-2': bleu2*100, 
       'BLEU-3': bleu3*100,
       'BLEU-4': bleu4*100,
       'METEOR': meteor*100,
       'ROUGE-L': rouge_l*100,
       'CIDEr-D': cider_score*100
   }