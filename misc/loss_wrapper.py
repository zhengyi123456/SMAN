import torch
import misc.utils as utils
import numpy as np
from misc.rewards import init_scorer, get_score
class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        self.crit = utils.LanguageModelCriterion()
        self.rl_crit = utils.RewardCriterion()

    def forward(self, fc_feats, att_feats, c3d_feats, labels, masks, att_masks, c3d_masks, gts, gt_indices, sc_flag):
        out = {}
        if not sc_flag:
            loss = self.crit(self.model(c3d_feats, att_feats, labels), labels[:, 1:], masks[:, 1:], self.opt.caption_model)
        else:
            self.model.eval()
            with torch.no_grad():
                greedy_res, _, greedy_res_layer1, _, greedy_res_layer2, _, greedy_res_layer3, _ = self.model(c3d_feats, att_feats, mode='sample')
            self.model.train()
            gen_result, sample_logprobs, gen_result_layer1, sample_logprobs_layer1, gen_result_layer2, sample_logprobs_layer2, gen_result_layer3, sample_logprobs_layer3\
                = self.model(c3d_feats, att_feats, opt={'sample_method':'sample'}, mode='sample')
            gts = [gts[_] for _ in gt_indices.tolist()]

            if self.opt.caption_model == 'all_gru':
                """LAYER 1"""
                scores_greedy_res_layer1 = get_score(greedy_res_layer1, gts, self.opt)
                scores_gen_result_layer1 = get_score(gen_result_layer1, gts, self.opt)
                scores_layer1 = scores_gen_result_layer1 - scores_greedy_res_layer1
                reward_layer1 = np.repeat(scores_layer1[:, np.newaxis], gen_result.shape[1], 1)
                reward_layer1 = torch.from_numpy(reward_layer1).float().cuda()
                loss_layer1, mask_layer1 = self.rl_crit(sample_logprobs_layer1, gen_result.data, reward_layer1)
                """LAYER 2"""
                scores_greedy_res_layer2 = get_score(greedy_res_layer2, gts, self.opt)
                scores_gen_result_layer2 = get_score(gen_result_layer2, gts, self.opt)
                scores_layer2 = scores_gen_result_layer2 - scores_greedy_res_layer2 + scores_gen_result_layer2 - scores_gen_result_layer1
                reward_layer2 = np.repeat(scores_layer2[:, np.newaxis], gen_result.shape[1], 1)
                reward_layer2 = torch.from_numpy(reward_layer2).float().cuda()
                loss_layer2, mask_layer2 = self.rl_crit(sample_logprobs_layer2, gen_result.data, reward_layer2)
                """LAYER 3"""
                scores_greedy_res_layer3 = get_score(greedy_res_layer3, gts, self.opt)
                scores_gen_result_layer3 = get_score(gen_result_layer3, gts, self.opt)
                scores_layer3 = scores_gen_result_layer3 - scores_greedy_res_layer3 + scores_gen_result_layer3 - scores_gen_result_layer2
                reward_layer3 = np.repeat(scores_layer3[:, np.newaxis], gen_result.shape[1], 1)
                reward_layer3 = torch.from_numpy(reward_layer3).float().cuda()
                loss_layer3, mask_layer3 = self.rl_crit(sample_logprobs_layer3, gen_result.data, reward_layer3)

                loss = torch.cat([loss_layer1, loss_layer2, loss_layer3], dim=-1)
                mask = torch.cat([mask_layer1, mask_layer2, mask_layer3], dim=-1)
                loss = torch.sum(loss) / torch.sum(mask)
                out['reward_layer1'] = reward_layer3[:, 0].mean()
                out['reward_layer2'] = reward_layer2[:, 0].mean()
                out['reward_layer3'] = reward_layer3[:, 0].mean()
                out['reward_layer4'] = reward_layer3[:, 0].mean()

                out['loss'] = loss
                return out

        out['loss'] = loss
        return out
