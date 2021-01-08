import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--learning_rate_decay_start', type=int, default=0)
    parser.add_argument('--learning_rate_decay_every', type=int, default=3)
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8)
    parser.add_argument('--reward_metric', type=str, default='cider')
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--acc_steps', type=int, default=1)
    parser.add_argument('--save_checkpoint_every', type=int, default=300)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--checkpoint_path', type=str, default='log/test')
    parser.add_argument('--val_split', type=str, default='test')
    parser.add_argument('--self_critical_after', type=int, default=-1)
    parser.add_argument('--start_from', type=str, default=None)
    parser.add_argument('--reduce_on_plateau', action='store_true')
    parser.add_argument('--caption_model', type=str, default="all_gru")
    parser.add_argument('--beam_size', type=int, default=2)
    parser.add_argument('--id', type=str, default='')
    # Data Input Settings
    parser.add_argument('--input_json', type=str, default='data/msvtttalk.json')
    parser.add_argument('--input_label_h5', type=str, default='data/msvtttalk_label.h5')
    parser.add_argument('--cached_tokens', type=str, default='data/msvtt-train-idxs')
    parser.add_argument('--input_box_feature', type=str, default='/data0/zy/msvtt/resnext/msvtt_box_features.h5')
    parser.add_argument('--input_c3d_feature', type=str, default='/data0/zy/msvtt/resnext/msvtt_c3d_features.h5')
    parser.add_argument('--input_app_feature', type=str, default='/data0/zy/msvtt/resnext/msvtt_appearance_features.h5')
    # Model Settings
    parser.add_argument('--rnn_size', type=int, default=1024)
    parser.add_argument('--input_encoding_size', type=int, default=512)
    parser.add_argument('--att_hid_size', type=int, default=512)
    parser.add_argument('--use_bn', type=int, default=2)
    # Optimization
    parser.add_argument('--grad_clip', type=float, default=0.1)
    parser.add_argument('--drop_prob_lm', type=float, default=0.5)
    parser.add_argument('--seq_per_img', type=int, default=40)
    parser.add_argument('--max_length', type=int, default=20)
    # Optimization: for the Language Model
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--optim_alpha', type=float, default=0.9)
    parser.add_argument('--optim_beta', type=float, default=0.999)
    parser.add_argument('--optim_epsilon', type=float, default=1e-8)
    args = parser.parse_args()
    return args

def add_eval_options(parser):
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--language_eval', type=int, default=1)
    parser.add_argument('--sample_method', type=str, default='greedy')
    parser.add_argument('--max_length', type=int, default=20, help='Maximum length during sampling')

    parser.add_argument('--input_json', type=str, default='data/msvtttalk.json')
    parser.add_argument('--input_label_h5', type=str, default='data/msvtttalk_label.h5')
    parser.add_argument('--cached_tokens', type=str, default='data/msvtt-train-idxs')
    parser.add_argument('--input_box_feature', type=str, default='/data0/zy/msvtt/resnext/msvtt_box_features.h5')
    parser.add_argument('--input_c3d_feature', type=str, default='/data0/zy/msvtt/resnext/msvtt_c3d_features.h5')
    parser.add_argument('--input_app_feature', type=str, default='/data0/zy/msvtt/resnext/msvtt_appearance_features.h5')

