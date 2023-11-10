import pandas as pd
import glob
from transformers import pipeline
import os
from nltk import sent_tokenize
from tqdm import tqdm
import os
import nltk
import warnings
from argparse import ArgumentParser


class TextClassifier:
    def __init__(self, model_checkpoint):
        self.token_classifier=pipeline(
            "token-classification",
            model=model_checkpoint,
            aggregation_strategy="simple",
            tokenizer=model_checkpoint,
            device=0
        )

    def classify_text(self, text):
        return self.token_classifier(text)


class ADUClassifier:
    def __init__(self, data_dir, output_dir, model_checkpoint, pred_col='preds'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.text_classifier = TextClassifier(model_checkpoint)
        self.pred_col = pred_col
        os.makedirs(self.output_dir, exist_ok=True)

    def classify(self):
        files = glob.glob(os.path.join(self.data_dir, '/**/*.jsonl'), recursive=True)

        for i, path in enumerate(tqdm(files)):
            df = pd.read_json(path, lines=True, orient='records')
            df['body'] = df['text'].apply(lambda x: sent_tokenize(x))
            df[self.pred_col] = [self.text_classifier.classify_text(text) for text in df['body']]
            df[self.pred_col] = df[self.pred_col].apply(lambda x: [item for sublist in x for item in sublist])
            df.to_json(os.path.join(self.output_dir, os.path.basename(path)), orient='records', lines=True)


def main(data_dir, output_dir, model_checkpoint):
    warnings.filterwarnings('ignore')
    nltk.download('punkt', quiet=True)

    data_processor = ADUClassifier(data_dir, output_dir, model_checkpoint, model_checkpoint)
    data_processor.classify()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, help='path to branches folder', default='../data/')
    parser.add_argument('-o', '--output_dir', type=str, default='../data/', help='path to output folder')
    parser.add_argument('-m', '--model_checkpoint', type=str, help='path to ADU identification model', required=True)
    parser.add_argument('-p', '--pred_col', type=str, default='preds', help='name of the column with predictions')
    args = parser.parse_args()

    main(args.data_dir, args.output_dir, args.model_checkpoint, args.tokenizer_model, args.pred_col)
