from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from .CaptionModel import CaptionModel
from models.allennlp_beamsearch import BeamSearch

class AttModel(CaptionModel):
    def __init__(self, opt):
        super(AttModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_hid_size = opt.att_hid_size
        self.gru_size = opt.rnn_size
        self.use_bn = getattr(opt, 'use_bn', 0)
        self.ss_prob = 0.
        self.opt = opt
        self.att_embed_flag = True
        """FEATURE EMBEDDING"""
        self.feature_size = 1024
        self.att_feat_size = 1536
        self.c3d_feat_size = 2048
        self.feat_size = self.rnn_size
        self.att_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(28 * 5),) if self.use_bn else ()) +
                (nn.Linear(self.att_feat_size, self.feat_size),
                 nn.ReLU(),
                 nn.Dropout(self.drop_prob_lm)) +
                ((nn.BatchNorm1d(28 * 5),) if self.use_bn == 2 else ())))
        self.c3d_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(28),) if self.use_bn else ()) +
                (nn.Linear(self.att_feat_size + self.c3d_feat_size, self.feat_size),
                 nn.ReLU(),
                 nn.Dropout(self.drop_prob_lm)) +
                ((nn.BatchNorm1d(28),) if self.use_bn == 2 else ())))
        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.c3d2att = nn.Linear(self.rnn_size, self.att_hid_size)
        """STATE INITIALIZATION"""
        self.h_output_proj = nn.Sequential(
            nn.Dropout(self.drop_prob_lm),
            nn.Linear(self.feature_size * 2, opt.rnn_size * 3),
            nn.ReLU(),
            nn.Dropout(self.drop_prob_lm))
        self.beam_search = BeamSearch(0, self.seq_length, 2, per_node_beam_size=opt.beam_size)

    def init_state(self, batch_size, init_state = None):
        if init_state is None:
            weight = next(self.parameters())
            return weight.new_zeros(batch_size, self.rnn_size), weight.new_zeros(batch_size, self.rnn_size), weight.new_zeros(batch_size, self.rnn_size)
        else:
            state = init_state.narrow(1, 0, self.rnn_size)
            att_out = init_state.narrow(1, self.rnn_size, self.rnn_size)
            sman_out = init_state.narrow(1, self.rnn_size * 2, self.rnn_size)
            return state, att_out, sman_out

    def clip_att(self, att_feats, att_masks):
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, fc_feats, att_feats):
        fc_feats = fc_feats.float()
        att_feats = att_feats.float()
        att_feats = self.att_embed(att_feats)
        fc_feats = self.c3d_embed(fc_feats)
        p_att_feats = self.ctx2att(att_feats)
        p_fc_feats = self.c3d2att(fc_feats)
        mean_att_feats = torch.mean(att_feats, dim=1)
        mean_fc_feats = torch.mean(fc_feats, dim=1)
        return mean_fc_feats, fc_feats, p_fc_feats, mean_att_feats, att_feats, p_att_feats

    # ---------------------------------------------------------------------#

    def masked_logprobs(self, fc_feats, output, batch_size, it):
        logprobs_masks = fc_feats.new_ones(batch_size, self.vocab_size + 1)
        logprobs_masks[:, output.size(1) - 1] = 0
        for bdash in range(batch_size):
            logprobs_masks[bdash, it[bdash]] = 0
        output = output.masked_fill(logprobs_masks == 0, -1e9)
        logprobs = F.log_softmax(output, dim=1)
        return logprobs

    def get_logprobs_state(self, it, mean_fc_feats, fc_feats, p_fc_feats, mean_att_feats, att_feats, p_att_feats, state, att_out, sman_out, motion_feats, visual_feats, text_feats, mode='train'):
        batch_size = fc_feats.size(0)
        """CORE FUNCTION"""
        output1, output2, output3, state, att_out, sman_out, motion_feat, visual_feat, text_feat = \
            self.core(it, mean_fc_feats, fc_feats, p_fc_feats, mean_att_feats, att_feats, p_att_feats, state, att_out, sman_out, motion_feats, visual_feats, text_feats)
        """LOGPROBS"""
        if mode == 'train':
            logprobs1 = F.log_softmax(output1, dim=1)
            logprobs2 = F.log_softmax(output2, dim=1)
            logprobs3 = F.log_softmax(output3, dim=1)
            logprobs = torch.cat([logprobs1.unsqueeze(1), logprobs2.unsqueeze(1), logprobs3.unsqueeze(1)], dim=1)
            return logprobs, state, att_out, sman_out, motion_feat, visual_feat, text_feat
        else:
            logprobs1 = self.masked_logprobs(fc_feats, output1, batch_size, it)
            logprobs2 = self.masked_logprobs(fc_feats, output2, batch_size, it)
            logprobs3 = self.masked_logprobs(fc_feats, output3, batch_size, it)
            """RETURN"""
            logprobs = torch.cat([logprobs1.unsqueeze(1), logprobs2.unsqueeze(1), logprobs3.unsqueeze(1)], dim=1)
            return logprobs, state, att_out, sman_out, motion_feat, visual_feat, text_feat

    # ---------------------------------------------------------------------#

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        batch_size = fc_feats.size(0)
        """FEATURE PREPARATION"""
        mean_fc_feats, fc_feats, p_fc_feats, mean_att_feats, att_feats, p_att_feats = \
            self._prepare_feature(fc_feats, att_feats)
        """OUTPUTS"""
        outputs = fc_feats.new_zeros(batch_size, 3, seq.size(1) - 1, self.vocab_size + 1)
        """FEATURE OUTPUTS"""
        motion_feats = fc_feats.new_zeros(batch_size, 0, self.rnn_size)
        visual_feats = fc_feats.new_zeros(batch_size, 0, self.rnn_size)
        text_feats = fc_feats.new_zeros(batch_size, 0, self.rnn_size)
        """STATE INITIALIZATION"""
        init_state_ = self.h_output_proj(torch.cat([mean_fc_feats, mean_att_feats], dim=-1))
        state, att_out, sman_out = self.init_state(batch_size, init_state_)
        """SEQUENCE PREDICTION"""
        for i in range(seq.size(1) - 1):
            it = seq[:, i].clone()
            """ENDING BREAK"""
            if i >= 1 and it.sum() == 0:
                break
            """CORE FUNCTION"""
            logprobs, state, att_out, sman_out, motion_feat, visual_feat, text_feat = \
                self.get_logprobs_state(it, mean_fc_feats, fc_feats, p_fc_feats, mean_att_feats, att_feats, p_att_feats, state, att_out, sman_out, motion_feats, visual_feats, text_feats)
            """FEATURE SAVE"""
            motion_feats = torch.cat([motion_feats, motion_feat.unsqueeze(1)], dim=1)
            visual_feats = torch.cat([visual_feats, visual_feat.unsqueeze(1)], dim=1)
            text_feats = torch.cat([text_feats, text_feat.unsqueeze(1)], dim=1)
            """OUTPUT SAVE"""
            outputs[:, :, i] = logprobs
        """RETURN"""
        return outputs

    def _sample(self, fc_feats, att_feats, opt={}):
        sample_method = opt.get('sample_method', 'greedy')
        temperature = opt.get('temperature', 1.0)
        batch_size = fc_feats.size(0)
        beam_size = opt.get('beam_size', 1)
        if beam_size > 1:
            print('BEAM SEARCH WITH BEAM SIZE: ', beam_size)
            return self._sample_beam(fc_feats, att_feats, opt)
        """FEATURE PREPARATION"""
        mean_fc_feats, fc_feats, p_fc_feats, mean_att_feats, att_feats, p_att_feats = self._prepare_feature(fc_feats, att_feats)
        """OUTPUTS"""
        seq_layer1 = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs_layer1 = fc_feats.new_zeros(batch_size, self.seq_length)
        seq_layer2 = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs_layer2 = fc_feats.new_zeros(batch_size, self.seq_length)
        seq_layer3 = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs_layer3 = fc_feats.new_zeros(batch_size, self.seq_length)
        seq_all = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs_all = fc_feats.new_zeros(batch_size, self.seq_length)
        """FEATURE OUTPUTS"""
        motion_feats = fc_feats.new_zeros(batch_size, 0, self.rnn_size)
        visual_feats = fc_feats.new_zeros(batch_size, 0, self.rnn_size)
        text_feats = fc_feats.new_zeros(batch_size, 0, self.rnn_size)
        """STATE INITIALIZATION"""
        init_state_ = self.h_output_proj(torch.cat([mean_fc_feats, mean_att_feats], dim=-1))
        state, att_out, sman_out = self.init_state(batch_size, init_state_)
        for t in range(self.seq_length):
            if t == 0:  # input <bos>
                it = fc_feats.new_zeros(batch_size, dtype=torch.long)
            """CORE FUNCTION"""
            logprobs, state, att_out, sman_out, motion_feat, visual_feat, text_feat = \
                self.get_logprobs_state(it, mean_fc_feats, fc_feats, p_fc_feats, mean_att_feats, att_feats, p_att_feats, state, att_out, sman_out, motion_feats, visual_feats, text_feats, mode='val')
            """THE TOTAL LOGPROBS"""
            logprobs_layer1 = logprobs[:, 0]
            logprobs_layer2 = logprobs[:, 1]
            logprobs_layer3 = logprobs[:, 2]
            logprobs_all = torch.mean(logprobs, dim=1)
            """SAMPLE"""
            it_layer1, sampleLogprobs_layer1 = self.sample_next_word(logprobs_layer1, sample_method, temperature)
            it_layer2, sampleLogprobs_layer2 = self.sample_next_word(logprobs_layer2, sample_method, temperature)
            it_layer3, sampleLogprobs_layer3 = self.sample_next_word(logprobs_layer3, sample_method, temperature)
            it_all, sampleLogprobs_all = self.sample_next_word(logprobs_all, sample_method, temperature)
            """UNFINISHED"""
            if t == 0:
                unfinished_all = it_all > 0
            else:
                unfinished_all = unfinished_all * (it_all > 0)
            """OUTPUT SAVE"""
            it_layer1 = it_layer1 * unfinished_all.type_as(it_layer1)
            seq_layer1[:, t] = it_layer1
            seqLogprobs_layer1[:, t] = sampleLogprobs_layer1.view(-1)
            it_layer2 = it_layer2 * unfinished_all.type_as(it_layer2)
            seq_layer2[:, t] = it_layer2
            seqLogprobs_layer2[:, t] = sampleLogprobs_layer2.view(-1)
            it_layer3 = it_layer3 * unfinished_all.type_as(it_layer3)
            seq_layer3[:, t] = it_layer3
            seqLogprobs_layer3[:, t] = sampleLogprobs_layer3.view(-1)
            it_all = it_all * unfinished_all.type_as(it_all)
            seq_all[:, t] = it_all
            seqLogprobs_all[:, t] = sampleLogprobs_all.view(-1)
            it = it_all.clone()
            """FEATURE SAVE"""
            motion_feats = torch.cat([motion_feats, motion_feat.unsqueeze(1)], dim=1)
            visual_feats = torch.cat([visual_feats, visual_feat.unsqueeze(1)], dim=1)
            text_feats = torch.cat([text_feats, text_feat.unsqueeze(1)], dim=1)
            """UNFINISHED BREAK"""
            if unfinished_all.sum() == 0:
                break
        """RETURN"""
        return seq_all, seqLogprobs_all, seq_layer1, seqLogprobs_layer1, seq_layer2, seqLogprobs_layer2, seq_layer3, seqLogprobs_layer3

    def _sample_beam(self, fc_feats, att_feats, opt={}):
        beam_size = opt.get('beam_size', 1)
        batch_size = fc_feats.size(0)
        self.batch_size = batch_size
        """FEATURE PREPARATION"""
        mean_fc_feats, fc_feats, p_fc_feats, mean_att_feats, att_feats, p_att_feats = self._prepare_feature(fc_feats, att_feats)
        """STATE INITIALIZATION"""
        init_state_ = self.h_output_proj(torch.cat([mean_fc_feats, mean_att_feats], dim=-1))
        state, att_out, sman_out = self.init_state(batch_size, init_state_)
        """FEATURE OUTPUTS"""
        motion_feats = fc_feats.new_zeros(batch_size, 0, self.rnn_size)
        visual_feats = fc_feats.new_zeros(batch_size, 0, self.rnn_size)
        text_feats = fc_feats.new_zeros(batch_size, 0, self.rnn_size)
        """START STATE FOR BEAM SEARCH"""
        start_state = {'state': state, 'att_out':att_out, 'sman_out':sman_out,
                       'mean_fc_feats': mean_fc_feats, 'fc_feats': fc_feats, 'p_fc_feats': p_fc_feats,
                       'mean_att_feats': mean_att_feats, 'att_feats': att_feats, 'p_att_feats': p_att_feats,
                       'motion_feats': motion_feats, 'visual_feats': visual_feats, 'text_feats': text_feats}
        it = fc_feats.new_zeros(batch_size, dtype=torch.long)
        predictions, log_prob = self.beam_search.search(it, start_state, self.beam_step)
        max_prob, max_index = torch.topk(log_prob, 1)  # b*1
        max_index = max_index.squeeze(1)  # b
        outputs = []
        for i in range(self.batch_size):
            outputs.append(predictions[i, max_index[i], :])
        outputs = torch.stack(outputs)
        return outputs, None, outputs, None, outputs, None, outputs, None

    def beam_step(self, last_predictions, current_state):
        group_size = last_predictions.size(0)  # batch_size or batch_size*beam_size
        batch_size = self.batch_size
        log_probs = []
        new_state = {}
        num = int(group_size / batch_size)  # 1 or beam_size
        for k, state in current_state.items():
            if isinstance(state, list):
                state = torch.stack(state, dim=1)
            _, *last_dims = state.size()
            current_state[k] = state.reshape(batch_size, num, *last_dims)
            new_state[k] = []
        for i in range(num):
            state = current_state['state'][:, i, :]
            att_out = current_state['att_out'][:, i, :]
            sman_out = current_state['sman_out'][:, i, :]
            mean_fc_feats = current_state['mean_fc_feats'][:, i, :]
            mean_att_feats = current_state['mean_att_feats'][:, i, :]
            fc_feats = current_state['fc_feats'][:, i, :]
            att_feats = current_state['att_feats'][:, i, :]
            p_fc_feats = current_state['p_fc_feats'][:, i, :]
            p_att_feats = current_state['p_att_feats'][:, i, :]
            motion_feats = current_state['motion_feats'][:, i, :]
            visual_feats = current_state['visual_feats'][:, i, :]
            text_feats = current_state['text_feats'][:, i, :]

            word_id = last_predictions.reshape(batch_size, -1)[:, i]

            logprobs, state, att_out, sman_out, motion_feat, visual_feat, text_feat = \
                self.get_logprobs_state(word_id, mean_fc_feats, fc_feats, p_fc_feats, mean_att_feats, att_feats, p_att_feats, state, att_out, sman_out, motion_feats, visual_feats, text_feats, mode='val')

            logprobs = torch.mean(logprobs, dim=1)

            motion_feats = torch.cat([motion_feats, motion_feat.unsqueeze(1)], dim=1)
            visual_feats = torch.cat([visual_feats, visual_feat.unsqueeze(1)], dim=1)
            text_feats = torch.cat([text_feats, text_feat.unsqueeze(1)], dim=1)

            log_probs.append(logprobs)

            new_state['state'].append(state)
            new_state['att_out'].append(att_out)
            new_state['sman_out'].append(sman_out)
            new_state['mean_fc_feats'].append(mean_fc_feats)
            new_state['mean_att_feats'].append(mean_att_feats)
            new_state['fc_feats'].append(fc_feats)
            new_state['att_feats'].append(att_feats)
            new_state['p_fc_feats'].append(p_fc_feats)
            new_state['p_att_feats'].append(p_att_feats)
            new_state['motion_feats'].append(motion_feats)
            new_state['visual_feats'].append(visual_feats)
            new_state['text_feats'].append(text_feats)

        log_probs = torch.stack(log_probs, dim=0).permute(1, 0, 2).reshape(group_size, -1)  # group_size*vocab_size
        # transform new state
        # from list to tensor(batch_size*beam_size, *)
        for k, state in new_state.items():
            new_state[k] = torch.stack(state, dim=0)  # (beam_size, batch_size, *)
            _, _, *last_dims = new_state[k].size()
            dim_size = len(new_state[k].size())
            dim_size = range(2, dim_size)
            new_state[k] = new_state[k].permute(1, 0, *dim_size)  # (batch_size, beam_size, *)
            new_state[k] = new_state[k].reshape(group_size, *last_dims)  # (batch_size*beam_size, *)
        return (log_probs, new_state)

class allGRUModel(AttModel):
    def __init__(self, opt):
        super(allGRUModel, self).__init__(opt)
        print('MODEL: {}'.format('allGRUModel'))
        self.core = Core(opt)

class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.att_hid_size = opt.att_hid_size
        self.rnn_size = opt.rnn_size
        self.h_embed = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, feats, p_feats, masks = None):
        feat_size = (feats.numel() // feats.size(0) // feats.size(-1))
        h_embed = self.h_embed(h)
        txt_replicate = h_embed.unsqueeze(1).expand_as(p_feats)
        hA = F.tanh(p_feats + txt_replicate)
        hAflat = self.alpha_net(hA.view(-1, self.att_hid_size))
        scores = hAflat.view(-1, feat_size)
        if masks is not None:
            scores = scores.masked_fill(masks == 0, -1e9)
        PI = F.softmax(scores, dim=1)
        visAtt = torch.bmm(PI.unsqueeze(1), feats)
        visAtt = visAtt.squeeze(1)
        return visAtt, PI

class Core(nn.Module):
    def __init__(self, opt):
        super(Core, self).__init__()
        self.opt = opt
        self.rnn_size = opt.rnn_size
        self.feature_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.input_encoding_size = opt.input_encoding_size
        """WORD EMBEDDING"""
        self.embed = nn.Sequential(
            nn.Embedding(opt.vocab_size + 1, opt.input_encoding_size),
            nn.ReLU(),
            nn.Dropout(self.drop_prob_lm))
        """LANGUAGE RNN"""
        self.gru_norm = nn.LayerNorm(self.rnn_size, 1e-5)
        self.gru = nn.GRUCell(self.input_encoding_size + self.rnn_size + self.feature_size * 2, self.rnn_size)
        """ATTENTION"""
        self.motion_attention = Attention(opt)
        self.visual_attention = Attention(opt)
        """ATTENTION GRU"""
        self.att_norm = nn.LayerNorm(self.rnn_size, 1e-5)
        self.att_gru = nn.GRUCell(self.rnn_size + self.feature_size * 2, self.rnn_size)
        """MOTION VISUAL CONTEXT"""
        self.visual2att = nn.Linear(opt.rnn_size, opt.att_hid_size)
        self.motion2att = nn.Linear(opt.rnn_size, opt.att_hid_size)
        self.text2att = nn.Linear(self.rnn_size, opt.att_hid_size)
        self.context_motion_attention = Attention(opt)
        self.context_visual_attention = Attention(opt)
        self.context_text_attention = Attention(opt)
        """SMAN GRU"""
        self.sman_norm = nn.LayerNorm(self.rnn_size, 1e-5)
        self.sman_gru = nn.GRUCell(self.rnn_size + self.feature_size * 2 + self.rnn_size, self.rnn_size)
        """OUTPUT"""
        self.logit1 = nn.Linear(opt.rnn_size, opt.vocab_size + 1)
        self.logit2 = nn.Linear(opt.rnn_size, opt.vocab_size + 1)
        self.logit3 = nn.Linear(opt.rnn_size, opt.vocab_size + 1)

    def forward(self, it, mean_fc_feats, fc_feats, p_fc_feats, mean_att_feats, att_feats, p_att_feats, state, att_out, sman_out, motion_feats, visual_feats, text_feats):
        batch_size = fc_feats.size(0)
        """LANGUAGE RNN"""
        xt = self.embed(it)
        gru_input = torch.cat([xt, mean_fc_feats, mean_att_feats, sman_out], dim=-1)
        h_output = self.gru(gru_input, state)
        h_output_norm = self.gru_norm(h_output)
        """ATTENTION"""
        h_motion, _ = self.motion_attention(h_output_norm, fc_feats, p_fc_feats)
        h_visual, _ = self.visual_attention(h_output_norm, att_feats, p_att_feats)
        """ATTENTION GRU"""
        att_gru_input = torch.cat([h_motion, h_visual, h_output_norm], dim=-1)
        att_out = self.att_gru(att_gru_input, att_out)
        att_out_norm = self.att_norm(att_out)
        """MOTION VISUAL CONTEXT"""
        visual_feats = torch.cat([visual_feats, h_visual.unsqueeze(1)], dim=1)
        motion_feats = torch.cat([motion_feats, h_motion.unsqueeze(1)], dim=1)
        text_feats = torch.cat([text_feats, h_output_norm.unsqueeze(1)], dim=1)
        p_visual_feats = self.visual2att(visual_feats)
        p_motion_feats = self.motion2att(motion_feats)
        p_text_feats = self.text2att(text_feats)
        context_motion_out, context_motion_out_map = self.context_motion_attention(att_out_norm, motion_feats, p_motion_feats)
        context_visual_out, context_visual_out_map = self.context_visual_attention(att_out_norm, visual_feats, p_visual_feats)
        context_text_out, context_text_out_map = self.context_text_attention(att_out_norm, text_feats, p_text_feats)
        """SMAN GRU"""
        sman_gru_input = torch.cat([context_motion_out, context_visual_out, context_text_out, att_out_norm], dim=-1)
        sman_out = self.sman_gru(sman_gru_input, sman_out)
        sman_out_norm = self.sman_norm(sman_out.unsqueeze(1)).squeeze(1)
        """OUTPUT"""
        h_pred_output = F.dropout(h_output_norm, self.drop_prob_lm, self.training)
        logit_h_output = self.logit1(h_pred_output)
        att_pred_output = F.dropout(att_out_norm, self.drop_prob_lm, self.training)
        logit_att_output = self.logit2(att_pred_output)
        sman_pred_output = F.dropout(sman_out_norm, self.drop_prob_lm, self.training)
        logit_sman_output = self.logit3(sman_pred_output)
        """RETURN"""
        return logit_h_output, logit_att_output, logit_sman_output, h_output_norm, att_out_norm, sman_out_norm, h_motion, h_visual, h_output_norm

