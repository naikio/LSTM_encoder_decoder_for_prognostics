import os
import sys
import datetime

import cmaps_dataset
from lstm_autoencoder_alpha import *
import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA


"""
Global config
"""
learning_rate = 0.001
hidden_n = 30
epochs = 10000
sequence_length = 20
batch_size = 80
n_components = 3    # n of principal components for PCA transform
reverse = False
use_pca = True
run_code = 'run_20180000__0000' # to set as: None for a new run, run_YYYYMMDD__hhmm to continue a previous run

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
    # new run: we create all the necessary folders
    run_code = datetime.datetime.now().strftime('%Y%m%d__%H%M')

    dir_run = 'run_' + run_code
    dir_log = dir_run + "/log"
    dir_model = dir_run + "/model"
    tf.gfile.MakeDirs(dir_run)
    tf.gfile.MakeDirs(dir_model)
    tf.gfile.MakeDirs(dir_log)
    print('*** RUN FOLDER : {}'.format(dir_run))
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
Dataset preparation
"""


def get_sequences_from_dataset(dataset, engine_labels, position=0):
    # dataset contains a sequence for each of the 100 engines
    # 80 engines are inside training set
    # 20 engines are inside validation set
    # this function takes 20 consecutive samples (which make a subsequence), starting from the position-th one

    # if parameter "position" is negative, take the n-th from last sequence

    output_set = np.array([]).reshape(0, sequence_length, n_components)
    for engine_number in np.unique(engine_labels):
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
        single_engine_sequence = np.expand_dims(single_engine_sequence, 0)
        output_set = np.vstack([output_set, single_engine_sequence])

    return output_set


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

    # Print total n of parameters
    print('%% Total parameters : ',
          np.sum([np.prod(dim) for dim in [variable.get_shape() for variable in tf.trainable_variables()]]))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:

        # restore model from disk
        if tf.gfile.Exists(os.path.join(dir_model, 'model.ckpt.meta')):
            print("Previous model found. Restoring...")
            saver.restore(sess, os.path.join(dir_model, 'model.ckpt'))
            print("Model Restored")
        else:
            print("No previous model found: starting from scratch")
            sess.run(tf.global_variables_initializer())

        # Instantiate SummaryWriters to output summaries and the Graph.
        train_writer = tf.summary.FileWriter(dir_log + '/train', sess.graph)
        train_writer_second = tf.summary.FileWriter(dir_log + '/train_second_seq', sess.graph)
        train_writer_middle = tf.summary.FileWriter(dir_log + '/train_middle_seq', sess.graph)
        train_writer_last = tf.summary.FileWriter(dir_log + '/train_last_seq', sess.graph)
        validation_writer_first = tf.summary.FileWriter(dir_log + '/validation_first_seq', sess.graph)

        # set some variables before starting the real work
        batches_per_epoch = int(80 / batch_size)
        epochs_since_save = 0   # keeps track of model savings
        epoch_of_last_save = 0
        last_saved_loss = sys.float_info.max
        for this_epoch in range(epochs):
            print("---------------------------------------------------------------")
            print("EPOCH %d" % this_epoch)
            print("---------------------------------------------------------------")

            training_set = get_sequences_from_dataset(train, train_engine_labels, position=0)

            for j in range(batches_per_epoch):
                train_batch = np.array([]).reshape(0, sequence_length, n_components)
                for b in range(batch_size):
                    index = j*batch_size+b
                    single_engine = training_set[index]
                    single_engine = np.expand_dims(single_engine, axis=0)
                    train_batch = np.vstack([train_batch, single_engine])
                if reverse:
                    # reverse the order of the sequence
                    train_batch_reverse = np.flip(train_batch, axis=1)
                else:
                    train_batch_reverse = train_batch

                feed_dict = {input_data: train_batch, input_data_reverse: train_batch_reverse}
                # the actual training step
                summary_str, _, loss_val = sess.run([model.summaries, model.training, model.loss], feed_dict=feed_dict)
                train_writer.add_summary(summary_str, this_epoch)
                if j % 20 == 0:
                    print("Step {}: --- Reconstruction error: {}".format(j, loss_val/batch_size))

            # Evaluate model against 2nd sequence
            test_batch_close = get_sequences_from_dataset(train, train_engine_labels, position=1)
            if reverse:
                # reverse the order of the sequence
                test_batch_close_reverse = np.flip(test_batch_close, axis=1)
            else:
                test_batch_close_reverse = test_batch_close
            feed_dict = {input_data: test_batch_close, input_data_reverse:test_batch_close_reverse}
            summary_str, loss_val = sess.run([model.summaries, model.loss], feed_dict=feed_dict)
            train_writer_second.add_summary(summary_str, this_epoch)
            print()
            print("Reconstruction error against 2nd sequence: {}".format(loss_val/batch_size))

            # Evaluate model against middle sequence
            test_batch_middle = get_sequences_from_dataset(train, train_engine_labels, position=108)
            if reverse:
                # reverse the order of the sequence
                test_batch_middle_reverse = np.flip(test_batch_middle, axis=1)
            else:
                test_batch_middle_reverse = test_batch_middle
            feed_dict = {input_data: test_batch_middle, input_data_reverse:test_batch_middle_reverse}
            summary_str, loss_val = sess.run([model.summaries, model.loss], feed_dict=feed_dict)
            train_writer_middle.add_summary(summary_str, this_epoch)
            print("Reconstruction error against 108th sequence: {}".format(loss_val/batch_size))

            # Evaluate model against last sequence
            test_batch_last = get_sequences_from_dataset(train, train_engine_labels, position=-1)
            if reverse:
                # reverse the order of the sequence
                test_batch_last_reverse = np.flip(test_batch_last, axis=1)
            else:
                test_batch_last_reverse = test_batch_last
            feed_dict = {input_data: test_batch_last, input_data_reverse:test_batch_last_reverse}
            summary_str, loss_val = sess.run([model.summaries, model.loss], feed_dict=feed_dict)
            train_writer_last.add_summary(summary_str, this_epoch)
            print("Reconstruction error against last sequence: {}".format(loss_val/batch_size))

            # Evaluate model against first sequence of validation sets
            validation_batch_first = get_sequences_from_dataset(validation, validation_engine_labels, position=0)
            if reverse:
                # reverse the order of the sequence
                validation_batch_first_reverse = np.flip(validation_batch_first, axis=1)
            else:
                validation_batch_first_reverse = validation_batch_first
            feed_dict = {input_data: validation_batch_first, input_data_reverse:validation_batch_first_reverse}
            summary_str, loss_val = sess.run([model.summaries, model.loss], feed_dict=feed_dict)
            validation_writer_first.add_summary(summary_str, this_epoch)
            print("Reconstruction error against first sequence (validation): {}".format(loss_val/validation_batch_first.shape[0]))
            print("Epoch of last save", epoch_of_last_save)
            print()

            epochs_since_save += 1

            # Save model each 250 epoch
            if this_epoch < 1500 and this_epoch % 250 == 0:
                # SAVE MODEL and update variables
                epochs_since_save, epoch_of_last_save, last_saved_loss = \
                    save_model_and_reset_epochs_count(this_epoch, loss_val)

            # Save model each 50 epoch and if better than previous
            if this_epoch > 1500 and epochs_since_save > 50 and loss_val < last_saved_loss:
                # SAVE MODEL and update variables
                epochs_since_save, epoch_of_last_save, last_saved_loss = \
                    save_model_and_reset_epochs_count(this_epoch, loss_val)

            # This check is for early stopping
            if epochs_since_save > 1000:
                print("1000 epochs without learning... Time to end this...")
                exit(10)
