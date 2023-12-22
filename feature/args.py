import argparse

def parser_args(mode="train"):

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",type=str, default="train.csv")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--model_dir", type=str, default='model/recent/')
    parser.add_argument("--kfold", type=int, default=2)
    parser.add_argument("--model", type=str, default='bert')
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--le_path", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    return args