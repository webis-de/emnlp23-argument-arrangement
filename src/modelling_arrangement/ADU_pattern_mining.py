import glob
import pandas as pd
import os
from tqdm import tqdm
from ast import literal_eval
from argparse import ArgumentParser

def remove_repetitions(lst):
    result = []
    for i in lst:
        if len(result) == 0 or i != result[-1]:
            result.append(i)
    return result

def get_seqs(data):
    sub_dict = {'Fact': 'F', 'Value': 'V', 'Policy': 'P', 'Testimony': 'T', 'Rhetorical': 'R'}
    seqs = [i['entity_group'] for i in literal_eval(data)]
    seqs = [sub_dict[i] for i in seqs]
    seqs = remove_repetitions(seqs)
    return str(seqs)

def main(data_path, op_file, comment_file):
    all_files = glob.glob(os.path.join(data_path, '**/**/*.jsonl'), recursive=True)
    for file in tqdm(all_files):
        df = pd.read_json(file, orient='records', lines=True)
        df['sequence'] = df['preds'].apply(get_seqs)
        df = df[['id', 'sequence']]
        op_df = df[df['id'].str.startswith('t3')]
        comment_df = df[df['id'].str.startswith('t1')]
        op_df.to_csv(f'{op_file}.csv', mode='a', header=False, index=False)
        comment_df.to_csv(f'{comment_file}.csv', mode='a', header=False, index=False)

    for filename in ['op_seqs.csv', 'comment_seqs.csv']:
        df = pd.read_csv(filename, names=['id', 'sequence'])
        df.drop_duplicates(subset=['id'], inplace=True)
        df.to_csv(filename, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-d', '--data-path', type=str, default='../data/', help='path to branches')
    parser.add_argument('-o', '--op-file', type=str, default='op_sequences', help='path to output file for OPs')
    parser.add_argument('-c', '--comment-file', type=str, default='comment_sequences', help='path to output file for comments')

    args = parser.parse_args()
    main(args.delta_path, args.non_delta_path)