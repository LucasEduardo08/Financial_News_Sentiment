import kagglehub
import pandas as pd
from datasets import load_dataset


# StockEmotios dataset links
PATH_TRAIN = "https://raw.githubusercontent.com/adlnlp/StockEmotions/refs/heads/main/tweet/train_stockemo.csv"
PATH_VAL = "https://raw.githubusercontent.com/adlnlp/StockEmotions/refs/heads/main/tweet/val_stockemo.csv"
PATH_TEST = "https://raw.githubusercontent.com/adlnlp/StockEmotions/refs/heads/main/tweet/test_stockemo.csv"

# TFN dataset links
SPLITS = {'train': 'sent_train.csv', 'validation': 'sent_valid.csv'}
PATH_TFN = "hf://datasets/zeroshot/twitter-financial-news-sentiment/"


def stockemotions_download():
    dataset_stockemotions_train = pd.read_csv(PATH_TRAIN, sep=',')
    dataset_stockemotions_val = pd.read_csv(PATH_VAL, sep=',')
    dataset_stockemotions_test = pd.read_csv(PATH_TEST, sep=',')

    # Saved datasets
    dataset_stockemotions_train.to_csv("data/train_stockemotions.csv", index=False)
    dataset_stockemotions_val.to_csv("data/val_stockemotions.csv", index=False)
    dataset_stockemotions_test.to_csv("data/test_stockemotions.csv", index=False)


def tfn_download():
    dataset_tfn_train = pd.read_csv(PATH_TFN + SPLITS['train'])
    dataset_tfn_val = pd.read_csv(PATH_TFN + SPLITS['validation'])

    # Save datasets
    dataset_tfn_train.to_csv("data/train_tfn.csv", index=False)
    dataset_tfn_val.to_csv("data/val_tfn.csv", index=False)


def main():
    stockemotions_download()
    tfn_download()

if __name__ == "__main__":
    main()

# Fazer commit e continuar com o desenvolvimento do código de pré-processamento dos dados, criação do modelo e treinamento.
