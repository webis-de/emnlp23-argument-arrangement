import pandas as pd
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from sklearn.utils import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
import nltk
import numpy as np
from get_data import DataCollector
from argparse import ArgumentParser
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=False)
nltk.download('stopwords', quiet=True)

class FeatureExtractor:
    def __init__(self, type, mode, path='../../data/'):
        self.type = type
        self.mode = mode
        self.path = path

    def get_splits(self):
        data_collector = DataCollector(self.path)
        train, test = data_collector.get_splits_for_mode(self.mode)
        return train, test

    def get_num_turns(self, paths):
        df = pd.DataFrame(columns=['id', 'length', 'label'])
        for path in tqdm(paths):
            length = len(pd.read_json(path, lines=True))
            id = path.split('/')[-1].split('.')[0]
            if 'non_delta' in path:
                label = 0
            else:
                label = 1
            data = {'id': id, 'length': length, 'label': label}
            df = pd.concat([df, pd.DataFrame(data, index=[0])], ignore_index=True)
        df = df.drop(columns=['id'])
        return df

    def get_words(self, data, role):
        data['words'] = data['text'].parallel_apply(lambda x: word_tokenize(x))
        if role == 'op':
            words = data.iloc[0]['words']
        elif role == 'rr':
            words = [item for sublist in data[1:]['words'].tolist() for item in sublist]
        return words

    def calculate_interplay(self, op, rr):
        int_int = 1. * len(set(op) & set(rr))
        if len(set(op)) == 0 or len(set(rr)) == 0:
            return [0, 0, 0, 0]
        return [int_int, int_int / len(set(rr)), int_int / len(set(op)), int_int / len(set(op) | set(rr))]

    def get_interplay(self, path, delta):
        from nltk.corpus import stopwords
        stopwords = set(stopwords.words('english'))
        data=pd.read_json(path, lines=True, orient='records')
        op = self.get_words(data, 'op')
        rr = self.get_words(data, 'rr')
        op_all = set(op)
        rr_all = set(rr)
        op_stop = op_all & stopwords
        rr_stop = rr_all & stopwords
        op_content = op_all - stopwords
        rr_content = rr_all - stopwords

        all_interplay = self.calculate_interplay(op_all, rr_all)
        stop_interplay = self.calculate_interplay(op_stop, rr_stop)
        content_interplay = self.calculate_interplay(op_content, rr_content)

        key = 'interplay'
        keys = [key + '_int', key + '_reply_frac', key + '_op_frac', key + '_jaccard']
        keys = ['all_' + i for i in keys] + ['stop_' + i for i in keys] + ['content_' + i for i in keys]
        d = dict(zip(keys, all_interplay + stop_interplay + content_interplay))
        if delta == True:
            d['label'] = 1
        elif delta == False:
            d['label'] = 0
        return d

    def get_interplay_for_split(self, split):
        df = pd.DataFrame()
        for path in tqdm(split):
            if 'non_delta' in path:
                delta = False
            else:
                delta = True
            df = pd.concat([df, pd.DataFrame(self.get_interplay(path, delta=delta), index=[0])], ignore_index=True)
        return df

    def get_features(self):
        train_paths, test_paths = self.get_splits()
        if self.type == 'interplay':
            df_train = self.get_interplay_for_split(train_paths)
            df_test = self.get_interplay_for_split(test_paths)
        elif self.type == 'length':
            df_train = self.get_num_turns(train_paths)
            df_test = self.get_num_turns(test_paths)
        return df_train, df_test

class LogisticRegressionModel:
    def __init__(self, train_features, test_features):
        self.model = LogisticRegression()
        self.train_features = train_features
        self.test_features = test_features

    def train(self):
        train_labels = self.train_features['label'].to_list()
        self.class_weights = compute_class_weight('balanced',
                                                  classes=np.unique(train_labels),
                                                  y=train_labels)
        self.class_weights = dict(zip(np.unique(train_labels), self.class_weights))
        self.model.class_weight = self.class_weights
        print(self.model.class_weight)
        self.model.fit(self.train_features.iloc[:, :-1], train_labels)

    def predict(self):
        test_labels = self.test_features['label'].to_list()
        self.predictions = self.model.predict(self.test_features.iloc[:, :-1])
        self.predictions_proba = self.model.predict_proba(self.test_features.iloc[:, :-1])
        self.auc = roc_auc_score(test_labels, self.predictions_proba[:, -1])
        self.classification_report = classification_report(test_labels, self.predictions)
        print(self.classification_report)
        print('ROC-AUC score: ', self.auc)

def main():
    parser=ArgumentParser()
    parser.add_argument('-d', '--path', type=str, default='../../data', help='path to the branches folder')
    parser.add_argument('-t', '--type', type=str, default='length', choices=['interplay', 'length'], help='interplay or length features')
    parser.add_argument('-m', '--mode', type=str, default='dialogue', choices=['dialogue', 'polylogue'], help='dialogue or polylogue branches')
    args=parser.parse_args()
    print('Features Type: ', args.type)
    print('Mode: ', args.mode)

    feature_extractor = FeatureExtractor(type=args.type, mode=args.mode, path=args.path)
    train_features, test_features = feature_extractor.get_features()
    model = LogisticRegressionModel(train_features, test_features)
    model.train()
    model.predict()


if __name__ == '__main__':
    main()