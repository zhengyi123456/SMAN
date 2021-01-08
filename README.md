## Stacked Multimodal Attention Network for Context-Aware Video Captioning
This repository includes the implementation for Stacked Multimodal Attention Network (SMAN) for Context-Aware Video Captioning.


#### Requierments
* Python 3.6.0
* PyTorch 1.1.0
* Java 1.8.0
* h5py 2.7.1


#### Data Preparation
The processed data have been provided [here](https://drive.google.com/drive/folders/1care9ZW3BRqLJ0G0O_BXD0YuPqE71J6F?usp=sharing). The processsed feature data for video have been provided [here](https://drive.google.com/drive/folders/1QvAwTmviFTqufwyucslnEVpvW_J0Br5J).

Download all required data, and the file directories should be like:
```
|-- coco-caption
|-- cider
|-- log
|   |-- allgru_rl
|   |   |-- infos_-best.pkl
|   |   |-- optimizer-best.pth
|   |   |-- model-best.pth
|-- data
|   |-- msrvtttalk.json
|   |-- msrvtttalk_label.h5
|   |-- msrvtt-train-words.p
|   |-- msrvtt-train-idxs.p
|   |-- dataset.json
|   |-- msrvtt_c3d_features.h5
|   |-- msrvtt_appearance_features.h5
|   |-- msrvtt_box_features.h5
```

#### Training
##### Training with Cross-Entropy Loss
```bash
python train.py --learning_rate 2e-4 --learning_rate_decay_start 0 --learning_rate_decay_every 2 --learning_rate_decay_rate 0.8 --max_epochs 10 --batch_size 10 --save_checkpoint_every 300 --checkpoint_path log/model --self_critical_after -1 --input_json data/msrvtttalk.json --input_label_h5 data/msrvtttalk_label.h5 --input_c3d_feature data/msrvtt_c3d_features.h5 --input_app_feature data/msrvtt_appearance_features.h5 --input_box_feature data/msrvtt_box_features.h5 --cached_tokens data/msrvtt-train-idxs 
```

##### Training with Self-Critical Loss
```bash
python train.py --learning_rate 2e-5 --learning_rate_decay_start -1 --max_epochs 40 --batch_size 10 --save_checkpoint_every 300 --checkpoint_path log --self_critical_after 0 --input_json data/msrvtttalk.json --input_label_h5 data/msrvtttalk_label.h5 --input_c3d_feature data/msrvtt_c3d_features.h5 --input_app_feature data/msrvtt_appearance_features.h5 --input_box_feature data/msrvtt_box_features.h5 --cached_tokens data/msrvtt-train-idxs --start_from log/model --reduce_on_plateau
```

#### Evaluation
```bash
python eval.py --model log/allgru_rl/model-best.pth --infos_path log/allgru_rl/infos_-best.pkl --input_json data/msrvtttalk.json --input_label_h5 data/msrvtttalk_label.h5 --input_c3d_feature data/msrvtt_c3d_features.h5 --input_app_feature data/msrvtt_appearance_features.h5 --input_box_feature data/msrvtt_box_features.h5 --cached_tokens data/msrvtt-train-idxs
```

BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | CIDEr | METEOR | ROUGE
:---: | :---: | :---: | :---: | :---: | :---: | :---: 
81.3|67.2|52.6|39.7|53.0|28.0|61.4

#### Acknowledgements

The implementation is based on [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch).
