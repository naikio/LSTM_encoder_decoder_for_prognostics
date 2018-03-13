import csv
import numpy as np


"""
Handles CMAPS Turbofan Engine Dataset
"""

"""
Dataset preparation
"""

def normalize_engines(dataset, engine_labels):

    unique_labels = np.unique(engine_labels)

    engines = [dataset[dataset[:,0] == i][:, 2:] for i in unique_labels] # split by engine
    engines_norm = []
    for eng in engines:
        std = np.std(eng, axis=0)
        m = np.mean(eng, axis=0)
        eng = (eng-m)/std
        eng = np.nan_to_num(eng)
        engines_norm.append(eng)

    engines_norm_flat = np.concatenate([e for e in engines_norm], axis=0)
    return engines_norm_flat

def get_test_from_FD00X_textfile(txt_fname):
    fd_engines = read_FD00X_textfile(txt_fname)

    test_engine_labels = fd_engines[:, 0]
    test_data = fd_engines

    return test_engine_labels, test_data

def get_train_and_validation_from_FD00X_textfile(txt_fname):
    fd_list_of_engines = read_FD00X_textfile(txt_fname)
    engine_train_labels, train, engine_validation_labels, validation = split_FD00X_train_validation(fd_list_of_engines,
                                                                                                                  random_state=0)
    return engine_train_labels, train, engine_validation_labels, validation

def read_FD00X_textfile(txt_fname):
    temp=[]
    with open(txt_fname, 'rt') as txt_file:
        lines = csv.reader(txt_file, delimiter=' ')
        for row in lines:
            if row:
                line = list(row)
                temp.append(np.asarray(line))
        temp = np.array(temp)
        temp[temp[:, :] == ''] = '0'  # convert to 0 all empty values
        temp = temp.astype(float)
        return temp

def split_FD00X_train_validation(fd_train, random_state=0, train_percentage=0.8):
    labels = list(np.unique(fd_train[:,0]).astype(int))
    np.random.RandomState(random_state).shuffle(labels)    # shuffle labels in-place
    split_idx = int(len(labels) * train_percentage)

    train_labels = labels[:split_idx]
    validation_labels = labels[split_idx:]

    train = fd_train[np.isin(fd_train[:, 0], train_labels)]
    validation = fd_train[np.isin(fd_train[:, 0], validation_labels)]

    train_labels = train[:, 0]
    validation_labels = validation[:, 0]

    return train_labels, train, validation_labels, validation


def get_sequences_from_dataset(dataset, engine_labels, sequence_length, n_components, position=0):
    # dataset contains a sequence for each of the 100 engines
    # 80 engines are inside training set
    # 20 engines are inside validation set
    # this function takes 20 consecutive samples (which make a subsequence), starting from the position-th one

    # if parameter "position" is negative, take the n-th from last sequence

    output_set = np.array([]).reshape(0, sequence_length, n_components)

    for engine_number in np.unique(engine_labels):
        single_engine_sequence = get_sequences_from_dataset_single_engine(dataset, engine_labels, engine_number, sequence_length, position)
        output_set = np.vstack([output_set, single_engine_sequence])

    return output_set

def get_sequences_from_dataset_single_engine(dataset, engine_labels, engine_number, sequence_length, position=0):

    # this function takes 20 consecutive samples (which make a subsequence), starting from the position-th one, FROM ONE ENGINE

    # if parameter "position" is negative, take the n-th from last sequence

    all_engine_seq = dataset[engine_labels == engine_number]
    # the code >>> array[R1 : R2]
    # take elements in range from R1 (INCLUDED) to R2 (EXCLUDED)
    if position < 0:
        if position == -1:
            single_engine_sequence = all_engine_seq[-sequence_length:]
        else:
            single_engine_sequence = all_engine_seq[position-sequence_length+1:position+1]
    else:
        single_engine_sequence = all_engine_seq[position:position + sequence_length]
    # single_engine_sequence = np.expand_dims(single_engine_sequence, 0)

    return single_engine_sequence