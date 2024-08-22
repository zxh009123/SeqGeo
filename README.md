This repo contains the Implementation and dataset of the paper ["Cross-View Image Sequence Geo-localization"](https://arxiv.org/abs/2210.14295).

### Prerequisite
---
```
Python 3
PyTorch > 1.10
TorchVision > 0.11
tqdm
numpy
```
### Training
---
```python
python train.py  --save_name SAVE_NAME --batch_size 24 --MHA_layers 6 --nHeads 8 --max_masked 6
```

### Test
---
```python
python test.py --model_folder SAVE_NAME
```

### Dataset
---
Please fill out [this](https://forms.gle/fSBJwmt1YgUqUVVh6) form to obtain access to the collected dataset. To be noticed, we will only share the data if it will be used for *RESEARCH PURPOSE* only. After downloading the dataset, please unzip and put all three folders (json, satellite, and street) under the dataset folder.

### Citation
---
```
@INPROCEEDINGS{10031017,
  author={Zhang, Xiaohan and Sultani, Waqas and Wshah, Safwan},
  booktitle={2023 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)}, 
  title={Cross-View Image Sequence Geo-localization}, 
  year={2023},
  volume={},
  number={},
  pages={2913-2922},
  keywords={Computer vision;Databases;Image sequences;Global Positioning System;Videos;Applications: Remote Sensing;Image recognition and understanding (object detection;categorization;segmentation;scene modeling;visual reasoning)},
  doi={10.1109/WACV56688.2023.00293}}
```
