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
