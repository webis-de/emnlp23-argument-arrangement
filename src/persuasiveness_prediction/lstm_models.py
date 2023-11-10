from tqdm import tqdm
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import warnings
import transformers
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from sklearn.utils import class_weight
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import get_data
import pandas as pd
from argparse import ArgumentParser
from sklearn.metrics import classification_report
transformers.logging.set_verbosity_error()
warnings.simplefilter(action='ignore')


class StrategiesDataset(Dataset):
    def __init__(self, data, bert_model='bert-large-uncased'):
        self.data = data
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BertModel.from_pretrained(bert_model).to(self.device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        texts = self.data.iloc[index]['text']
        label = self.data.iloc[index]['label']
        sequence = self.data.iloc[index]['sequence']

        input_ids, attention_mask = self.preprocess_texts(texts)
        embedded_texts = self.embed_texts(input_ids, attention_mask)
        sequence = torch.tensor(sequence, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return embedded_texts, sequence, label

    def preprocess_texts(self, texts):
        encoded_texts = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        input_ids = encoded_texts['input_ids']
        attention_mask = encoded_texts['attention_mask']
        return input_ids.to(self.device), attention_mask.to(self.device)

    def embed_texts(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.model(input_ids.to(self.device), attention_mask=attention_mask.to(self.device))
            embedded_texts = outputs.pooler_output
        return embedded_texts.to(self.device)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.0):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)
        return out


class CombinedClassifier(nn.Module):
    def __init__(self):
        super(CombinedClassifier, self).__init__()
        self.bert_classifier = LSTM(input_size=1024, hidden_size=256, num_layers=1, num_classes=2)
        self.structural_classifier = LSTM(input_size=1, hidden_size=32, num_layers=1, num_classes=2)
        self.softmax = nn.Softmax(dim=1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x, y):
        bert_output = self.bert_classifier(x)
        structural_output = self.structural_classifier(y)
        combined_output = bert_output.to(self.device) + structural_output.to(self.device)
        combined_output = self.softmax(combined_output)
        return combined_output.to(self.device)


def collate_fn(batch):
    sequences, clusters, labels = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=-100)
    clusters_padded = pad_sequence(clusters, batch_first=True, padding_value=-100)
    labels = torch.tensor(labels, dtype=torch.long)
    return sequences_padded, clusters_padded.unsqueeze(2), labels

def prepare_data(file_paths, cluster_type='sgt_cluster', bert_model='bert-large-uncased'):
    dataset = pd.DataFrame(columns=['text', 'sequence', 'label'])

    for file in tqdm(file_paths):
        df = pd.read_json(file, lines=True, orient='records')
        clusters = [x for x in df[cluster_type].tolist()]
        texts = df.text.tolist()
        if 'non_delta' in file:
            data = {'text': texts, 'sequence': clusters, 'label': 0}
        else:
            data = {'text': texts, 'sequence': clusters, 'label': 1}
        dataset = dataset.append(data, ignore_index=True)
    return StrategiesDataset(dataset, bert_model)


def train(model, model_type, train_loader, test_loader, epochs, lr, weight_decay, save_path='output'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_loss = np.inf

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            embedded_texts, clusters, labels = batch

            if model_type == 'combined':
                outputs = model(embedded_texts.to(device), clusters.to(device)).squeeze(0)
            elif model_type == 'bert':
                outputs = model(embedded_texts.to(device)).squeeze(0)
            elif model_type == 'lstm':
                outputs = model(clusters.to(device)).squeeze(0)

            loss = criterion(outputs.to(device), labels.to(device))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss /= len(train_loader)

        predictions = []
        true_labels = []
        with torch.no_grad():
            model.eval()
            val_loss = 0
            for batch in tqdm(test_loader):
                embedded_texts, clusters, labels = batch
                if model_type == 'combined':
                    outputs = model(embedded_texts.to(device), clusters.to(device)).squeeze(0)
                elif model_type == 'bert':
                    outputs = model(embedded_texts.to(device)).squeeze(0)
                elif model_type == 'lstm':
                    outputs = model(clusters.to(device)).squeeze(0)
                predictions += outputs.argmax(dim=1).tolist()
                true_labels += labels.tolist()
                loss = criterion(outputs.to(device), labels.to(device))
                val_loss += loss.item()
            val_loss /= len(test_loader)

        print(f'Epoch: {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(classification_report(true_labels, predictions))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print('Saved model')
    return model

def main():
    parser = ArgumentParser()
    parser.add_argument('--data_path', default='../../data')
    parser.add_argument('--mode', choices=['dialogue', 'polylogue'], default='dialogue')
    parser.add_argument('--model_type', choices=['bert', 'lstm', 'combined'], default='combined')
    parser.add_argument('--cluster_type', choices=['cluster_sgt', 'cluster_edit'], default='cluster_sgt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--save_path', default='output')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    data_collector = get_data.DataCollector(path=args.data_path)
    train_files, test_files = data_collector.get_splits_for_mode(args.mode)
    print(f'\nTraining on {len(train_files)} files, testing on {len(test_files)} files')

    train_dataset = prepare_data(train_files, args.cluster_type)
    test_dataset = prepare_data(test_files, args.cluster_type)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    if args.model_type == 'bert':
        model = LSTM(input_dim=1024, hidden_dim=256, output_dim=2, num_layers=1).to(device)
    elif args.model_type == 'lstm':
        model = LSTM(input_size=1, hidden_size=32, num_layers=1, num_classes=2).to(device)
    elif args.model_type == 'combined':
        model = CombinedClassifier()
    model = train(model, args.model_type, train_loader, test_loader, args.epochs, args.lr, args.weight_decay, args.save_path)

if __name__ == '__main__':
    main()