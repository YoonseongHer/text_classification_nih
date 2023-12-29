import pandas as pd
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer
import os
import matplotlib.pyplot as plt
import numpy as np
import itertools
import warnings
from sklearn.metrics import classification_report, confusion_matrix
from feature.model import test_model, tokenizer_fit_model
from feature.dataloader import Custom_Dataset
from feature.preprocessing import data_preprocessing
from feature.args import parser_args
from feature.util import plot_confusion_matrix
import torch

warnings.filterwarnings(action="ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

args = parser_args()
device = args.device
model = torch.load(args.model_path,
                   map_location=device)
tokenizer = tokenizer_fit_model(model)
data = pd.read_csv(args.data_dir)

with open(args.le_path,'rb') as f:
    le = pickle.load(f).tolist()
    
data, label_class = data_preprocessing(data,le)
test_loader = Custom_Dataset(data,tokenizer).data_loader(512,1,1)
flat_true_labels, flat_predictions = test_model(model, test_loader, device)
