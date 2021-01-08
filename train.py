from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import traceback
import sys
import opts
import models
from dataloader import *
import eval_utils
import misc.utils as utils
from misc.rewards import init_scorer
from misc.loss_wrapper import LossWrapper
import os

os.environ["CUDA_VISIBLE_DEVICES"]='0'

try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

def train(opt):
    acc_steps = getattr(opt, 'acc_steps', 1)

    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length
    opt.ix_to_word = loader.ix_to_word

    tb_summary_writer = tb and tb.SummaryWriter(opt.checkpoint_path)

    infos = {}
    histories = {}
    if opt.start_from is not None:
        with open(os.path.join(opt.start_from, 'infos_' + opt.id + '.pkl'), 'rb') as f:
            infos = utils.pickle_load(f)
            saved_model_opt = infos['opt']
            need_be_same = ["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme
        if os.path.isfile(os.path.join(opt.start_from, 'histories_' + opt.id + '.pkl')):
            with open(os.path.join(opt.start_from, 'histories_' + opt.id + '.pkl'), 'rb') as f:
                histories = utils.pickle_load(f)
    else:
        infos['iter'] = 0
        infos['epoch'] = 0
        infos['iterators'] = loader.iterators
        infos['split_ix'] = loader.split_ix
        infos['vocab'] = loader.get_vocab()
    infos['opt'] = opt

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    opt.vocab = loader.get_vocab()
    model = models.setup(opt).cuda()
    del opt.vocab
    dp_model = torch.nn.DataParallel(model)
    lw_model = LossWrapper(model, opt)
    dp_lw_model = torch.nn.DataParallel(lw_model)

    epoch_done = True
    # Assure in training mode
    dp_lw_model.train()

    if opt.noamopt:
        optimizer = utils.get_std_opt(model, factor=opt.noamopt_factor, warmup=opt.noamopt_warmup)
        optimizer._step = iteration
    elif opt.reduce_on_plateau:
        optimizer = utils.build_optimizer(model.parameters(), opt)
        optimizer = utils.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    else:
        optimizer = utils.build_optimizer(model.parameters(), opt)

    def save_checkpoint(model, infos, optimizer, histories=None, append=''):
        if len(append) > 0:
            append = '-' + append
        # if checkpoint_path doesn't exist
        if not os.path.isdir(opt.checkpoint_path):
            os.makedirs(opt.checkpoint_path)
        checkpoint_path = os.path.join(opt.checkpoint_path, 'model%s.pth' %(append))
        torch.save(model.state_dict(), checkpoint_path)
        print("model saved to {}".format(checkpoint_path))
        optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer%s.pth' %(append))
        torch.save(optimizer.state_dict(), optimizer_path)
        with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'%s.pkl' %(append)), 'wb') as f:
            utils.pickle_dump(infos, f)
        if histories:
            with open(os.path.join(opt.checkpoint_path, 'histories_'+opt.id+'%s.pkl' %(append)), 'wb') as f:
                utils.pickle_dump(histories, f)

    try:
        while True:
            sys.stdout.flush()
            if epoch_done:
                if not opt.noamopt and not opt.reduce_on_plateau:
                    # Assign the learning rate
                    if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                        frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                        decay_factor = opt.learning_rate_decay_rate ** frac
                        opt.current_lr = opt.learning_rate * decay_factor
                    else:
                        opt.current_lr = opt.learning_rate
                    utils.set_lr(optimizer, opt.current_lr)
                    print('Learning Rate: ', opt.current_lr)
                if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                    sc_flag = True
                    init_scorer(opt.cached_tokens)
                else:
                    sc_flag = False
                epoch_done = False

            data = loader.get_batch('train')
            if (iteration % acc_steps == 0):
                optimizer.zero_grad()

            torch.cuda.synchronize()
            start = time.time()
            tmp = [data['fc_feats'], data['att_feats'], data['c3d_feats'], data['labels'], data['masks'], data['att_masks'], data['c3d_masks']]
            tmp = [_ if _ is None else _.cuda() for _ in tmp]
            fc_feats, att_feats, c3d_feats, labels, masks, att_masks, c3d_masks = tmp

            model_out = dp_lw_model(fc_feats, att_feats, c3d_feats, labels, masks, att_masks, c3d_masks, data['gts'], torch.arange(0, len(data['gts'])), sc_flag)

            loss = model_out['loss'].mean()
            loss_sp = loss / acc_steps

            loss_sp.backward()
            if ((iteration + 1) % acc_steps == 0):
                utils.clip_gradient(optimizer, opt.grad_clip)
                optimizer.step()
            torch.cuda.synchronize()
            train_loss = loss.item()
            end = time.time()
            if iteration % 1 == 0:
                if not sc_flag:
                    print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}".format(iteration, epoch, train_loss, end - start))
                else:
                    print("iter {} (epoch {}), reward1 = {:.3f}, reward2 = {:.3f}, reward3 = {:.3f}, train_loss = {:.3f}, time/batch = {:.3f}".format(iteration, epoch, model_out['reward_layer1'].mean(), model_out['reward_layer2'].mean(), model_out['reward_layer3'].mean(), train_loss, end - start))

            iteration += 1
            if data['bounds']['wrapped']:
                epoch += 1
                epoch_done = True

            if (iteration % opt.losses_log_every == 0):
                add_summary_value(tb_summary_writer, 'train_loss', train_loss, iteration)
                if opt.noamopt:
                    opt.current_lr = optimizer.rate()
                elif opt.reduce_on_plateau:
                    opt.current_lr = optimizer.current_lr
                add_summary_value(tb_summary_writer, 'learning_rate', opt.current_lr, iteration)
                add_summary_value(tb_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
                if sc_flag:
                    add_summary_value(tb_summary_writer, 'reward1', model_out['reward_layer1'].mean(), iteration)
                    add_summary_value(tb_summary_writer, 'reward2', model_out['reward_layer2'].mean(), iteration)
                    add_summary_value(tb_summary_writer, 'reward3', model_out['reward_layer3'].mean(), iteration)

                loss_history[iteration] = train_loss
                lr_history[iteration] = opt.current_lr
                ss_prob_history[iteration] = model.ss_prob

            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['iterators'] = loader.iterators
            infos['split_ix'] = loader.split_ix

            if (iteration % opt.save_checkpoint_every == 0):
                # eval model
                eval_kwargs = {'split': opt.val_split, 'dataset': opt.input_json}
                eval_kwargs.update(vars(opt))
                val_loss, predictions, lang_stats = eval_utils.eval_split(dp_model, lw_model.crit, loader, eval_kwargs)
                print('Summary Epoch {} Iteration {}: CIDEr: {} BLEU-4: {}'.format(epoch, iteration, lang_stats['CIDEr'], lang_stats['Bleu_4']))

                if opt.reduce_on_plateau:
                    if opt.reward_metric == 'cider':
                        optimizer.scheduler_step(-lang_stats['CIDEr'])
                    elif opt.reward_metric == 'bleu':
                        optimizer.scheduler_step(-lang_stats['Bleu_4'])
                    elif opt.reward_metric == 'meteor':
                        optimizer.scheduler_step(-lang_stats['METEOR'])
                    elif opt.reward_metric == 'rouge':
                        optimizer.scheduler_step(-lang_stats['ROUGE_L'])
                    else:
                        optimizer.scheduler_step(val_loss)
                # Write validation result into summary
                add_summary_value(tb_summary_writer, 'validation loss', val_loss, iteration)
                if lang_stats is not None:
                    for k,v in lang_stats.items():
                        add_summary_value(tb_summary_writer, k, v, iteration)
                val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

                # Save model if is improving on validation result
                if opt.language_eval == 1:
                    if opt.reward_metric == 'cider':
                        current_score = lang_stats['CIDEr']
                    elif opt.reward_metric == 'bleu':
                        current_score = lang_stats['Bleu_4']
                    elif opt.reward_metric == 'meteor':
                        current_score = lang_stats['METEOR']
                    elif opt.reward_metric == 'rouge':
                        current_score = lang_stats['ROUGE_L']
                else:
                    current_score = - val_loss

                best_flag = False

                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True

                # Dump miscalleous informations
                infos['best_val_score'] = best_val_score
                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['lr_history'] = lr_history
                histories['ss_prob_history'] = ss_prob_history

                save_checkpoint(model, infos, optimizer, histories)
                if opt.save_history_ckpt:
                    save_checkpoint(model, infos, optimizer, append=str(iteration))

                if best_flag:
                    save_checkpoint(model, infos, optimizer, append='best')

            # Stop if reaching max epochs
            if epoch >= opt.max_epochs and opt.max_epochs != -1:
                break

    except (RuntimeError, KeyboardInterrupt):
        print('Save ckpt on exception ...')
        save_checkpoint(model, infos, optimizer)
        print('Save ckpt done.')
        stack_trace = traceback.format_exc()
        print(stack_trace)

if __name__ == '__main__':
    opt = opts.parse_opt()
    train(opt)