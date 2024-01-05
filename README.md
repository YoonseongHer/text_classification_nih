# 보건관련 text 자료 분류 

<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=PyTorch&logoColor=white"/>&nbsp;<img src="https://img.shields.io/badge/Python-3776AB?style=square&logo=Python&logoColor=white"/>&nbsp;[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20face-yellow)](https://huggingface.co/models?filter=keytotext)&nbsp;
## 사용방법
### Requirements
    python == 3.9.16
    torch == 1.10.1
    transformers == 4.31.0
    
### TRAIN model
    python3 train.py --data_dir [data_dir] --kfold [fold_num] --model_dir [model_save_dir] --model [model type]

### TEST model
    python3 test.py --model_path [model_dir] --device [device] --data_dir [data_dir] --le_path [labelencoder_dir] --save_dir [result_save_dir]

### predict model
    python3 predict.py --model_path [model_dir] --device [device] --data_dir [data_dir] --le_path [labelencoder_dir] --save_dir [result_save_dir]
