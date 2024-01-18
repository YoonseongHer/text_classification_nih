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
from torch.multiprocessing import freeze_support

if __name__=='__main__':
    freeze_support()
    warnings.filterwarnings(action="ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = parser_args()
    os.makedirs(args.save_dir, exist_ok=True)
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
    print(classification_report(flat_predictions, flat_true_labels, digits=4))
    mc = confusion_matrix(y_true=flat_true_labels, y_pred=flat_predictions)
    plot_confusion_matrix(cm           = mc, 
                          normalize    = False,
                          target_names = le,
                          title        = "Confusion Matrix",
                          save_dir= args.save_dir)
    data['predict'] = flat_predictions
    data['label'] = data['label'].apply(lambda x: le[x])
    data['predict'] = data['predict'].apply(lambda x: le[x])
    data.to_csv(args.save_dir+'test_result.csv',index=False)