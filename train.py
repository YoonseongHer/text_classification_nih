import pandas as pd
import os
import warnings
import pickle
from sklearn.model_selection import StratifiedKFold
from feature.args import parser_args
from feature.model import import_model_tokenizer, train_model
from feature.dataloader import Custom_Dataset
from feature.preprocessing import data_preprocessing
from torch.multiprocessing import freeze_support

if __name__=='__main__':
    freeze_support()
    warnings.filterwarnings(action="ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = parser_args()
    os.makedirs(args.model_dir,exist_ok=True)
    data = pd.read_csv(args.data_dir)
    data, label_class = data_preprocessing(data)

    model, tokenizer = import_model_tokenizer(len(label_class),args.model)

    kfold = StratifiedKFold(n_splits=args.kfold)
    for fold, (train_ids, val_ids) in enumerate(kfold.split(data, data['label'].values)):
        print("")
        print('='*30 + ' Fold {:} / {:} '.format(fold + 1, kfold.n_splits) + '='*30)
        train_data, val_data = data.loc[train_ids], data.loc[val_ids]
        train_loader = Custom_Dataset(train_data,tokenizer).data_loader(args.max_len,
                                                                        args.batch_size,
                                                                        args.num_workers)
        val_loader = Custom_Dataset(val_data,tokenizer).data_loader(args.max_len,
                                                                    args.batch_size,
                                                                    args.num_workers)
        train_model(model,
                    train_loader,
                    val_loader,
                    args.lr,
                    args.epochs,
                    args.device,
                    args.model_dir+f'fold-{fold+1}-')

    with open(args.model_dir+'label_class.pkl', 'wb') as f:
        pickle.dump(label_class, f)