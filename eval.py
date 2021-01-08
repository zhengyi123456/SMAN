from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import opts
import models
from dataloader import *
from dataloaderraw import *
import eval_utils
import argparse
import misc.utils as utils
import torch

os.environ["CUDA_VISIBLE_DEVICES"]='0'

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='log/allgru_rl/model-best.pth')
parser.add_argument('--infos_path', type=str, default='log/allgru_rl/infos_-best.pkl')
parser.add_argument('--beam_size', type=int, default=2)

opts.add_eval_options(parser)
opt = parser.parse_args()

with open(opt.infos_path, 'rb') as f:
    infos = utils.pickle_load(f)

replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 'input_json', 'batch_size']
ignore = ['start_from']

for k in vars(infos['opt']).keys():
    if k in replace:
        setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
    elif k not in ignore:
        if not k in vars(opt):
            vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

vocab = infos['vocab']
opt.vocab = vocab
model = models.setup(opt)
del opt.vocab
model.load_state_dict(torch.load(opt.model))
model.cuda()
model.eval()
crit = utils.LanguageModelCriterion()
loader = DataLoader(opt)

# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.
loader.ix_to_word = infos['vocab']

opt.datset = opt.input_json
loss, split_predictions, lang_stats = eval_utils.eval_split(model, crit, loader,  vars(opt))

for k, v in lang_stats.items():
    print('{}: {}'.format(k, v))

with open('results.json', 'w') as f:
    json.dump(split_predictions, f)