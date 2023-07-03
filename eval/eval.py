import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("./tokenizer")

import nltk
import numpy as np
from nltk import ngrams
from collections import Counter
import operator
from scipy import stats
import json
import os
from multiset_distances import MultisetDistances
import mauve
from multiprocessing.pool import Pool
from functools import partial
import torch

def zipf(cands):
    cnt = Counter()
    for tokens in cands:
        cnt.update(tokens)

    xs = np.arange(1, min(len(cnt), 5000)+1)
    ys = np.array(sorted(cnt.values(), key=operator.neg)[:5000])
    a, b, r, p, std = stats.linregress(np.log(xs), np.log(ys))
    return {"zipf": -a}

def bleu(refs, cands):
    result = {}
    for i in range(1, 5):
        result["corpus-bleu-%d"%i] = "%.4f"%(nltk.translate.bleu_score.corpus_bleu(list_of_references=[[r] for r in refs], hypotheses=cands, weights=tuple([1./i for j in range(i)])))
    for i in range(1, 5):
        result["r-corpus-bleu-%d"%i] = "%.4f"%(nltk.translate.bleu_score.corpus_bleu(list_of_references=[[c] for c in cands], hypotheses=refs, weights=tuple([1./i for j in range(i)])))
    for i in range(1, 5):
        result["sent-bleu-%d"%i] = []
        for r, c in zip(refs, cands):
            result["sent-bleu-%d"%i].append(nltk.translate.bleu_score.sentence_bleu(references=[r], hypothesis=c, weights=tuple([1./i for j in range(i)])))
        result["sent-bleu-%d"%i] = "%.4f"%np.mean(result["sent-bleu-%d"%i])
    return result

def mauve_score(refs, cands, device):
    input_refs = refs
    input_cands = cands
    score = {}
    model_path = "/data/guanjian/transformers_model/gpt2-medium"
    out = mauve.compute_mauve(p_text=input_refs, q_text=input_cands, device_id=int(device[-1]), max_text_length=512, featurize_model_name=model_path)
    score["mauve_score"] = out.mauve
    return score

def msj(refs, cands):
    ref_avg_len = 0
    hyp_avg_len = 0
    for line in refs:
        ref_avg_len += len(line)
    ref_avg_len /= len(refs)
    for line in cands:
        hyp_avg_len += len(line)
    hyp_avg_len /= len(cands)

    msd = MultisetDistances(references=refs, min_n=1, max_n=5)
    msj_distance = msd.get_jaccard_score(sentences=cands)
    tmp_result = {}
    for k in msj_distance:
        tmp_result["msj-%d"%k] = msj_distance[k]
    return tmp_result

def distinct(name, cands):
    result = {}
    for i in range(1, 6):
        all_ngram, all_ngram_num = {}, 0.
        for k, cand in enumerate(cands):
            ngs = ["_".join(c) for c in ngrams(cand, i)]
            all_ngram_num += len(ngs)
            for s in ngs:
                if s in all_ngram:
                    all_ngram[s] += 1
                else:
                    all_ngram[s] = 1
        result["distinct-%d"%i] = "%.4f"%(len(all_ngram) / float(all_ngram_num))
    # fout.close()
    return result

def length(cands, name):
    length = []
    for c in cands:
        length.append(len(c))
    return {"length": "%.4f"%np.mean(length)}

def tok_repeat_l(cands, device):
    metrics = {}
    for c_len in [16,32,128]:
        metrics.update({f"tok_repeat_{c_len}": []})

    for k, hyp in enumerate(cands):
        if k % 1000 == 0:
            print("processing %d lines"%k)
        hyp_id = tokenizer.convert_tokens_to_ids(hyp)
        for c_len in [16,32,128]:
            hypo = torch.tensor(hyp_id).long().to(device)
            T = hypo.size(0)
            
            prev_hypo = hypo.expand(T, T).masked_fill(torch.ones(T, T).triu().bool().to(device), -1)

            prev_hypo = prev_hypo.masked_fill(torch.ones(T, T).tril(-c_len-1).bool().to(device), -1)

            repeat = (hypo[:, None] == prev_hypo)
            has_repeat = repeat.sum(1).gt(0).float()[1:]
            metrics[f"tok_repeat_{c_len}"] += has_repeat.cpu().numpy().tolist()
    for k, v in metrics.items():
        metrics[k] = "%.4f"%float(np.mean(v))
    return metrics



def get_result(name, truth, cand):
    result = {}
    truth_token  = [tokenizer.tokenize(t)[:512] for t in truth]
    cand_token = [tokenizer.tokenize(c)[:512] for c in cand]

    result.update(mauve_score(truth, cand, device=device))
    result.update(bleu(truth_token, cand_token))
    result.update(msj(truth_token, cand_token))
    result.update(distinct(name, cand_token))
    # result.update(tok_repeat_l(cand_token, device=device))
    result.update(length(cand_token, name))
    result.update(zipf(cand_token))
    key = sorted(result.keys())
    key = result.keys()
    for k in key:
        print(name, k, result[k])
    print("="*10)
    return result

device = "cuda:5"
result_list = [
    "../result/wikitext_selfcont_output_greedy.txt",
    "../result/wikitext_selfcont_output_topp06.txt",
]
with open("../wikitext_data/test.txt", "r", encoding="utf-8") as fin:
    truth = [line.strip() for line in fin][:1000]

for name in result_list:
    print(name)
    cand = []
    with open(name, "r", encoding="utf-8") as fin:
        for line in fin:
            cand.append(line.strip())
    result = get_result(name, truth, cand[:1000])