from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import torch
from .allGRUModel import allGRUModel

def setup(opt):
    if opt.caption_model == 'all_gru':
        model = allGRUModel(opt)
    else:
        raise Exception("Caption model not supported: {}".format(opt.caption_model))

    if vars(opt).get('start_from', None) is not None:
        # check if all necessary files exist
        assert os.path.isdir(opt.start_from)," %s must be a a path" % opt.start_from
        assert os.path.isfile(os.path.join(opt.start_from,"infos_"+opt.id+".pkl")),"infos.pkl file does not exist in path %s"%opt.start_from
        print('LOADING THE MODEL FROM {}'.format(opt.start_from))
        model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model.pth')))

    return model