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
Will be released soon