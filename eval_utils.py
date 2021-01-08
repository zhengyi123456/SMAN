from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils

bad_endings = ['a','an','the','in','for','at','of','with','before','after','on','upon','near','to','is','are','am']
bad_endings += ['the']

def count_bad(sen):
    sen = sen.split(' ')
    if sen[-1] in bad_endings:
        return 1
    else:
        return 0

def language_eval(dataset, preds, model_id, split, checkpoint_path):
    import sys
    sys.path.append("coco-caption")
    annFile = 'coco-caption/annotations/captions_val_msvtt.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap
    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join(checkpoint_path, 'cache_'+ model_id + '_' + split + '.json')
    coco = COCO(annFile)
    valids = coco.getImgIds()
    only_valids = []
    preds_filt = []
    for pred in preds:
        if pred['image_id'] in valids and pred['image_id'] not in only_valids:
            preds_filt.append(pred)
            only_valids.append(pred['image_id'])

    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w'))  # serialize to temporary json file. Sigh, COCO API...
    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()
    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score
    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
    out['bad_count_rate'] = sum([count_bad(_['caption']) for _ in preds_filt]) / float(len(preds_filt))
    outfile_path = os.path.join('eval_results/', model_id + '_' + split + '.json')
    with open(outfile_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)
    return out

def eval_split(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'test')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    remove_bad_endings = eval_kwargs.get('remove_bad_endings', 0)
    os.environ["REMOVE_BAD_ENDINGS"] = str(remove_bad_endings)  # Use this nasty way to make other code clean since it's a global configuration
    model.eval()
    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []

    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size
        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['c3d_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data['att_masks'] is not None else None,
               data['c3d_masks'][np.arange(loader.batch_size) * loader.seq_per_img]]
        tmp = [_.cuda() if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, c3d_feats, att_masks, c3d_masks = tmp
        with torch.no_grad():
            seq, _, seq1, _, seq2, _, seq3, _ = model(c3d_feats, att_feats, opt=eval_kwargs, mode='sample')
        sents = utils.decode_sequence(loader.get_vocab(), seq)
        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            predictions.append(entry)
            if verbose:
                print('image %s: %s' % (entry['image_id'], entry['caption']))
                print('------------------------------------------------------')

        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' % (ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, eval_kwargs['checkpoint_path'].split('/')[-1], split, "")

    # Switch back to training mode
    model.train()
    return loss_sum / loss_evals, predictions, lang_stats