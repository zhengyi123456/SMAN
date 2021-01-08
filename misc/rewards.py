from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import time
import misc.utils as utils
from collections import OrderedDict
import torch
import sys
sys.path.append("cider")
from pyciderevalcap.ciderD.ciderD import CiderD
sys.path.append("coco-caption")
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
CiderD_scorer = None
Bleu_scorer = None
Meteor_scorer = None
Rouge_scorer = None

def init_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)
    global Bleu_scorer
    Bleu_scorer = Bleu_scorer or Bleu(4)
    global Meteor_scorer
    Meteor_scorer = Meteor()
    global Rouge_scorer
    Rouge_scorer = Rouge()

def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()

def get_score(gen_result, data_gts, opt):
    batch_size = gen_result.size(0)  # batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data_gts)
    res = OrderedDict()
    gen_result = gen_result.data.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]
    res_ = [{'image_id': i, 'caption': res[i]} for i in range(batch_size)]
    res__ = {i: res[i] for i in range(batch_size)}
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(batch_size)}
    reward_metric = opt.reward_metric

    if reward_metric == 'cider':
        _, cider_scores = CiderD_scorer.compute_score(gts, res_)
        scores = cider_scores

    elif reward_metric == 'bleu':
        _, bleu_scores = Bleu_scorer.compute_score(gts, res__)
        bleu_scores = np.array(bleu_scores[3])
        scores = bleu_scores

    elif reward_metric == 'meteor':
        _, meteor_scores = Meteor_scorer.compute_score(gts, res__)
        meteor_scores = np.array(meteor_scores)
        scores = meteor_scores

    elif reward_metric == 'rouge':
        _, rouge_scores = Rouge_scorer.compute_score(gts, res__)
        rouge_scores = np.array(rouge_scores)
        scores = rouge_scores

    return scores