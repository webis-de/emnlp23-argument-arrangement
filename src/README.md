## Modelling Argument Arrangement 

This projects contains the following scripts:

1.  `ADU_identification.py` - applies the ADU type classifier to the dataset and saves the predictions in the field of each comment. 
Model checkpoint can be found on [huggingface]().


| Argument | Default    | Description                      |
|----------|------------|----------------------------------|
| `-d`     | `../data/` | path to the main data folder     |
| `-o`     | `../data/` | path to the output folder        |
| `-m`     |            | path to the model checkpoint     |
| `-p`     | `preds`    | column name for the model output |

2. `ADU_pattern_mining.py` - mines and abstracts sequences of ADU types from the dataset.

| Argument | Default             | Description                                                |
|----------|---------------------|------------------------------------------------------------|
| `-d`     | `../data/`          | path to the main data folder                               |
| `-o`     | `op_sequences`      | name of the csv file to store unique OP ADU sequences      |
| `-c`     | `comment_sequences` | name of the csv file to store unique comment ADU sequences |

3. `ADU_clustering.py` - applies hierarchical stustering to the sequences of ADU types using the SGT and edit distance clustering approaches. 
Output file contains the unique comment id and the cluster ID for the chosen clustering approach.

| Argument | Default                   | Options        | Description                  |
|----------|---------------------------|----------------|------------------------------|
| `-d`     | `../data/`                |                | path to the main data folder |
| `-c`     | `sgt`                     | `[sgt, edits]` | clustering approach          |
| `-n`     | `10`                      |                | number of output clusters    |
| `-o`     | `clustered_sequences.csv` |                | name of the output file      |


## Persuasion Prediction

This projects contains the following scripts:

1. `regression_models.py` - code for training and evaluating the 'length based classifier' and 'interplay features classifier' models.

| Argument | Default       | Options                 | Description                  |
|----------|---------------|-------------------------|------------------------------|
| `-d`     | `../../data/` |                         | path to the main data folder |
| `-t`     | `length`      | `[length, interplay]`   | feature type                 |
| `-m`     | `dialogue`    | `[dialogue, polylogue]` | discussion mode/scenario     |

2. `lstm_models.py` - code for training and evaluating the LSTM-based models.

| Argument         | Default       | Options                       | Description                           |
|------------------|---------------|-------------------------------|---------------------------------------|
| `--data_path`    | `../../data/` |                               | path to the main data folder          |
| `--mode`         | `dialogue`    | `[dialogue, polylogue]`       | discussion mode/scenario              |
| `--model_type`   | `combined`    | `[bert, lstm, combined]`      | model type                            |
| `--cluster_type` | `cluster_sgt` | `[cluster_sgt, cluster_edit]` | which cluster type to use as features |
| `--epochs`       | `10`          |                               | number of epochs                      |
| `--batch_size`   | `32`          |                               | batch size                            |
| `--weight_decay` | `0.0001`      |                               | weight decay                          |
| `--lr`           | `0.0001`      |                               | learning rate                         |
| `--save_path`    | `output`      |                               | output path for trained model         |


