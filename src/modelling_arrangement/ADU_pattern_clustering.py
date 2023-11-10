from sklearn.cluster import AgglomerativeClustering
from sgt import SGT
from ast import literal_eval
import numpy as np
import pandas as pd
import distance
from argparse import ArgumentParser

class DataProcessor:
    '''
    DataProcessor class
    - Reads in the data
    - Converts the sequences into the format required by SGT/Edits
    '''
    def __init__(self, data_path=None):
        self.data_path = data_path

    def process(self, data_path):
        self.df = pd.read_csv(data_path)
        self.df['sequence'] = self.df['sequence'].apply(literal_eval)
        self.df['joined'] = self.df['sequence'].apply(lambda x: ''.join(x))
        return self.df


class DistanceCalculator:
    '''
    DistanceCalculator class
    - Calculates the distance matrix for clustering
    '''
    def __init__(self, df):
        self.df = df

    def normalized_levenshtein_distance(self, str1, str2):
        '''
        Normalized Levenshtein Distance
        Params:
            str1: string 1
            str2: string 2
        Returns:
            Levenshtein Distance normalized by the length of the longer string
        '''
        _distance=distance.levenshtein(str1, str2)
        normalized_distance=_distance / max(len(str1), len(str2))
        return normalized_distance

    def edit_distance(self):
        '''
        Levenshtein Distance
        - It requires negative similarities, so -1 * levenshtein(t1, t2)
        '''
        strings = np.asarray(self.df['joined'].to_list(), dtype=object)
        _distance = np.array([[self.normalized_levenshtein_distance(list(w1), list(w2)) for w1 in strings] for w2 in strings])
        return np.asarray(_distance)

    def sgt_embeddings(self, kappa=1, flatten=True, lengthsensitive=False, mode='default'):
        sgt = SGT(kappa=kappa, flatten=flatten, lengthsensitive=lengthsensitive, mode=mode)
        embedding = sgt.fit_transform(self.df[['id', 'sequence']])
        embedding = embedding.set_index('id')
        embedding.dropna(inplace=True)
        return embedding


class Clusterer:
    '''
    Clusterer class
    - Clusters the data given the distance matrix
    '''
    def __init__(self, df, distance_matrix, n_clusters):
        self.df = df
        self.distance_matrix = distance_matrix
        self.n_clusters = n_clusters


    def cluster(self, type):
        '''
        Clusters the data
        Params:
            type: 'sgt' or 'edits'
        Returns:
            df: dataframe with cluster labels
        '''
        model = AgglomerativeClustering(n_clusters=self.n_clusters, affinity='euclidean', linkage='ward')
        model.fit(self.distance_matrix)
        self.df[type] = model.labels_
        return self.df

def main(data_path, cluster_type, n_clusters, out_path):
    '''
    Main function
    Params:
        data_path: path to the file with ADU type sequences
        cluster_type: 'sgt' or 'edits'
        n_clusters: number of clusters
        out_path: path to save the output
    Returns:
        df: dataframe with cluster labels
    '''
    processor = DataProcessor()
    df = processor.process(data_path)
    distance_calculator = DistanceCalculator(df)
    if cluster_type == 'sgt':
        distance_matrix = distance_calculator.sgt_embeddings()
    elif cluster_type == 'edits':
        distance_matrix = distance_calculator.edit_distance()
    clusterer = Clusterer(df, distance_matrix, n_clusters)
    df = clusterer.cluster(cluster_type)
    df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, default='op_seqs.csv')
    parser.add_argument('-c', '--cluster_type', type=str, default='sgt', choices=['sgt', 'edits'])
    parser.add_argument('-n', '--n_clusters', type=int, default=10)
    parser.add_argument('-o', '--out_path', type=str, default='clustered_sequences.csv')
    args = parser.parse_args()
    main(**vars(args))