import glob
from sklearn.model_selection import train_test_split
import os

class DataCollector:
    def __init__(self, path='../branches'):
        self.path=path
        self.data=self.get_data()

    def get_data(self):
        '''
        Recursively search for all jsonl files in the branches folder and split them into train and test sets
        Params:
            path: str, path to the branches folder
        Returns:
            data: dict of train and test sets for each mode (dialogue and polylogue)
        '''

        modes=['dialogue', 'polylogue']
        data={}

        for mode in modes:
            delta = sorted(glob.glob(os.path.join(self.path, 'delta', mode, '*.jsonl')))
            non_delta = sorted(glob.glob(os.path.join(self.path, 'non_delta', mode, '*.jsonl')))

            delta_train, delta_test=train_test_split(delta, test_size=0.2, random_state=42, shuffle=False)
            non_delta_train, non_delta_test=train_test_split(non_delta, test_size=0.2, random_state=42, shuffle=False)

            data[f'{mode}']={
                'delta_train': delta_train,
                'delta_test': delta_test,
                'non_delta_train': non_delta_train,
                'non_delta_test': non_delta_test
            }

        return data

    def get_splits_for_mode(self, mode='dialogue'):
        '''
        Get train and test splits for a given mode (dialogue or polylogue)
        Params:
            mode: str, dialogue or polylogue
        Returns:
            train: list of train files
            test: list of test files
        '''

        data = self.data[mode]
        train = data['delta_train'] + data['non_delta_train']
        test = data['delta_test'] + data['non_delta_test']
        return train, test
