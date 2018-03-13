import os
from sklearn.decomposition import PCA
from lstm_autoencoder_alpha import *

import datetime

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from sklearn.externals import joblib
import cmaps_dataset


"""
Global config
"""
learning_rate = 0.001
hidden_n = 30
epochs = 10000
sequence_length = 20
batch_size = 80
n_components = 3    # n of principal components for PCA transform
reverse = True
use_pca = True
run_code = 'run_20180000__0000'


if not use_pca:
    n_components = 26

"""
Directories
"""
#at each run, create a folder
#   . run_YYYYMMDD_hhmm
#   |----model
#       |----model checkpoints
#   |----log
#       |----log files for tensorboard

if run_code is None:
    # abort - this is not the training script, we NEED a pretrained model
    dir_model = None
    print("Run train.py first to train a model")
    exit(100)
else:
    # not a new run: we just delete LOG folder. Later, we will also load previous model from dir_model
    dir_run = run_code
    dir_log = dir_run + "/log"
    dir_model = dir_run + "/model"

    if tf.gfile.Exists(dir_log):
        tf.gfile.DeleteRecursively(dir_log)
    print('*** CONTINUE EXISTING RUN')
    print('*** RUN FOLDER : {}'.format(dir_run))


"""
Model utilities
"""
def save_model_and_reset_epochs_count(e, l):
    save_path = saver.save(sess, os.path.join(dir_model, 'model.ckpt'))
    print("####################################################################################")
    print("######################################################      Model saved in file: %s" % save_path)
    print("####################################################################################")

    return 0, e, l

"""
Other Utilities
"""
def moving_average(x, N=20):
    return np.convolve(x, np.ones(N,)/N, mode='valid')
# def moving_average(a, n=15):
#     ret = np.cumsum(a, dtype=float)
#     ret[n:] = ret[n:] - ret[:-n]
#     return ret[n - 1:] / n

# def moving_average(x, N=15):
#     cumsum = np.cumsum(np.insert(x, 0, 0))
#     return (cumsum[N:] - cumsum[:-N]) / float(N)



# keep same random seed across different runs = 0
rnd = np.random.RandomState(0)

# C-MAPSS TURBOFAN DATASET
# we select only file number 1: FD001
dataset_folder = "../dataset/CMAPSSData/"
engine = "FD001"

train_txt_fname = dataset_folder + "train_" + engine + ".txt"
test_txt_fname = dataset_folder + "test_" + engine + ".txt"
rul_txt_fname = dataset_folder + "RUL_" + engine + ".txt"

train_engine_labels, train_data, validation_engine_labels, validation_data = cmaps_dataset.get_train_and_validation_from_FD00X_textfile(train_txt_fname)
test_engine_labels, test_data = cmaps_dataset.get_test_from_FD00X_textfile(test_txt_fname)

train = cmaps_dataset.normalize_engines(train_data, train_engine_labels)
validation = cmaps_dataset.normalize_engines(validation_data, validation_engine_labels)
test = cmaps_dataset.normalize_engines(test_data, test_engine_labels)
# PCA
X = np.concatenate((train, validation))
if use_pca:
    pca = PCA(n_components=n_components)
    pca.fit(X)
    train = pca.transform(train)
    validation = pca.transform(validation)
    test = pca.transform(test)

##############################################
# Model
##############################################
with tf.Graph().as_default():
    input_data = tf.placeholder(tf.float32, [None, sequence_length, train.shape[1]])
    input_data_reverse = tf.placeholder(tf.float32, [None, sequence_length, train.shape[1]])

    model = LSTMAutoencoder(hidden_n=hidden_n, inputs=input_data, inputs_reverse=input_data_reverse)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    if not os.path.isfile('e_curves_train.pkl'):

        with tf.Session(config=config) as sess:

            # restore model from disk
            if tf.gfile.Exists(os.path.join(dir_model, 'model.ckpt.meta')):
                print("Previous model found. Restoring...")
                saver.restore(sess, os.path.join(dir_model, 'model.ckpt'))
                print("Model Restored")
            else:
                print("No previous model found: starting from scratch")
                sess.run(tf.global_variables_initializer())


            e_curves_train = []
            original_pts_train = []
            unique_labels, counts = np.unique(train_engine_labels, return_counts=True)
            tic = datetime.datetime.now()
            for i, eng_number, l in zip(range(len(unique_labels)), unique_labels, counts):
                e_curve = []
                reco_seq_stacked = []
                print('\rEngine {} - computing errors...'.format(int(eng_number)), end='', flush=True)
                for j in range(l-sequence_length+1):
                    empty_seq = np.full(shape=(l, n_components), fill_value=np.nan)
                    sequence = cmaps_dataset.get_sequences_from_dataset_single_engine(train, train_engine_labels,
                                                                                      engine_number=eng_number, sequence_length=sequence_length, position=j)
                    sequence = sequence.reshape((-1, sequence_length, n_components))
                    # sequence_reverse = np.flip(sequence, axis=1)
                    feed_dict = {input_data:sequence, input_data_reverse:sequence}
                    loss, reco_seq= sess.run(fetches=[model.loss, model.inference], feed_dict=feed_dict)
                    empty_seq[j:j + sequence_length] = reco_seq
                    e_curve.append(loss)
                    reco_seq_stacked.append(empty_seq)
                reco_pts = np.nanmean(reco_seq_stacked, axis=0)
                original_pts = cmaps_dataset.get_sequences_from_dataset_single_engine(train, train_engine_labels, engine_number=eng_number, sequence_length=5000, position=0)
                e_curve = np.average(((reco_pts-original_pts)**2), axis=1)
                print('\rEngine {} - done - {}%'.format(int(eng_number), (i+1)/len(unique_labels)*100))
                e_curves_train.append(e_curve)
                original_pts_train.append(original_pts)
            toc = datetime.datetime.now()
            joblib.dump(e_curves_train, "e_curves_train.pkl")
            joblib.dump(original_pts_train, "original_pts_train.pkl")
            print('Elapsed time: {} seconds', (toc-tic).seconds)

            e_curves_test = []
            original_pts_test = []
            unique_labels, counts = np.unique(test_engine_labels, return_counts=True)
            tic = datetime.datetime.now()
            for i, eng_number, l in zip(range(len(unique_labels)), unique_labels, counts):
                e_curve = []
                reco_seq_stacked = []
                print('\rEngine {} - computing errors...'.format(int(eng_number)), end='', flush=True)
                for j in range(l-sequence_length+1):
                    empty_seq = np.full(shape=(l, n_components), fill_value=np.nan)
                    sequence = cmaps_dataset.get_sequences_from_dataset_single_engine(test, test_engine_labels,
                                                                                      engine_number=eng_number, sequence_length=sequence_length, position=j)
                    sequence = sequence.reshape((-1, sequence_length, n_components))
                    # sequence_reverse = np.flip(sequence, axis=1)
                    feed_dict = {input_data:sequence, input_data_reverse:sequence}
                    loss, reco_seq= sess.run(fetches=[model.loss, model.inference], feed_dict=feed_dict)
                    empty_seq[j:j + sequence_length] = reco_seq
                    e_curve.append(loss)
                    reco_seq_stacked.append(empty_seq)
                reco_pts = np.nanmean(reco_seq_stacked, axis=0)
                original_pts = cmaps_dataset.get_sequences_from_dataset_single_engine(test, test_engine_labels, engine_number=eng_number, sequence_length=5000, position=0)
                e_curve = np.average(((reco_pts-original_pts)**2), axis=1)
                print('\rEngine {} - done - {}%'.format(int(eng_number), (i+1)/len(unique_labels)*100))
                e_curves_test.append(e_curve)
                original_pts_test.append(original_pts)
            toc = datetime.datetime.now()
            joblib.dump(e_curves_test, "e_curves_test.pkl")
            joblib.dump(original_pts_test, "original_pts_test.pkl")
            print('Elapsed time: {} seconds', (toc-tic).seconds)

    else:
        e_curves_train = joblib.load('e_curves_train.pkl')
        e_curves_test = joblib.load('e_curves_test.pkl')
        original_pts_train = joblib.load('original_pts_train.pkl')
        original_pts_test = joblib.load('original_pts_test.pkl')

    hi_curve_all_engines = []
    for l in e_curves_train:
        e_max = np.max(l)
        e_min = np.min(l)

        hi_curve = []
        for e in l:
            hi = (e_max - e) / (e_max - e_min)
            hi_curve.append(hi)
        hi_curve_all_engines.append(hi_curve)

    joblib.dump(hi_curve_all_engines, 'hi_curves_train.pkl')


    for l in e_curves_train:
        plt.plot(l)
    plt.show()

    for l in hi_curve_all_engines:
        N=20
        ls = np.convolve(l, np.ones((N,))/N, mode='same')
        lv = np.convolve(l, np.ones((N,))/N, mode='valid')
        lf = np.convolve(l, np.ones((N,))/N, mode='full')
        plt.plot(l)
        #plt.plot(ls)
        plt.plot(lv)
        #plt.plot(lf)
        plt.ylim((0, 1))
        plt.show()

    for l in hi_curve_all_engines:
        plt.plot(l)
        plt.ylim((0, 1))
    plt.show()