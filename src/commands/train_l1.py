#!/usr/bin/python
import sys
import os
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import argparse
from data import prepare_l1_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import pickle as pk
from models import *
import tensorflow as tf
import h5py
import gc
from scoring import *
import pandas as pd
import multiprocessing
from data import DataGenerator, describe_features
from utils import PlotL1RUL
import logging

logging.basicConfig(level=logging.INFO)



FEATURES = ['alt', 'Mach', 'TRA', 'T2', 'T24', 'T30', 'T48', 'T50', 'P15', 'P2',
           'P21', 'P24', 'Ps30', 'P40', 'P50', 'Nf', 'Nc', 'Wf', 'Fc', 'hs']

           
def train(seed, W, batch_size, epochs, hyper_params, cache_dir, net_summary=True): 
        logging.info("Starting training for fold with seed %d" % seed)
        
        # crate the model
        model = create_cnn_model((20,W,1), **hyper_params)
        if net_summary:
            for l in str(model.summary()).split('\n'):
                logging.info(l)
        
        # read fold data sets
        train_set_file_path = os.path.join(cache_dir, 'train_l1_%d.h5' % seed)
        test_set_file_path = os.path.join(cache_dir, 'test_l1_%d.h5' % seed)
        
        
        # create data generators
        extra_channel = True
        X_train = pd.read_hdf(train_set_file_path, key='phm21')
        logging.info("Feature stats of the train set:")
        describe_features(X_train, FEATURES)
        train_gen = DataGenerator(X_train, FEATURES, window_size=W, 
                                      batch_size=batch_size, 
                                      epoch_len_reducer=1000, 
                                      add_extra_channel=extra_channel)
        del X_train
        
        X_test = pd.read_hdf(test_set_file_path, key='phm21')
        logging.info("\n\n---------------------------------------------------------------------")
        logging.info("Feature stats of the test set:")
        describe_features(X_test, FEATURES)
        test_gen = DataGenerator(X_test, FEATURES, window_size=W, 
                                     batch_size=256, epoch_len_reducer=1000, 
                                     add_extra_channel=extra_channel)
        del X_test

        
        train_gen.epoch_len_reducer = 2000
        train_gen.return_label = True
        
        test_gen.epoch_len_reducer = 400
        test_gen.return_label = True

        # plot rul data
        ids = list(test_gen._X.keys())[:4]
        X_rul = {_id:test_gen._X[_id] for _id in ids}
        Y_rul = {_id:test_gen._Y[_id] for _id in ids}
        
        # train
        es = tf.keras.callbacks.EarlyStopping(monitor='val_Score', patience=8)
        rlr = tf.keras.callbacks.ReduceLROnPlateau(patience=3)
        pr = PlotL1RUL(X_rul, Y_rul, W)
        history = model.fit(train_gen, validation_data=test_gen, 
                            epochs=epochs, verbose=2, 
                            callbacks=[es, rlr, pr])
        
        history_path = os.path.join(cache_dir, 'cnn_history_%d.pk' % seed)
        pk.dump(history.history, open(history_path, 'wb'))
        model_path = os.path.join(cache_dir, 'cnn_l1_%d.h5' % seed)
        model.save(model_path)
            
        
CNN_HYPER_PARAMS = {
    'block_size': 4.41106, 
    'conv_activation': 1.69354, 
    'dense_activation': 1.28829, 
    'dilation_rate': 1.839, 
    'dropout': 0.131517, 
    'fc1': 74.786, 
    'fc2': 100, 
    'kernel_size': 0.688211, 
    'l1': 0.000723271, 
    'l2': 0, 
    'lr': 0.001, 
    'nblocks': 4.02792, 
}

BATCH_SIZE = 63
W = 162
EPOCHS = 100

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare data for L1 model training')
    args = parser.parse_args()
    cache_dir = prepare_l1_data()
    
    for seed in [999, 666, 128, 256, 394]:  
        model_path = os.path.join(cache_dir, 'cnn_l1_%d.h5' % seed)
        
        if not os.path.exists(model_path):
            queue = multiprocessing.Queue()
            p = multiprocessing.Process(target=train, args=(seed, W, BATCH_SIZE, EPOCHS, CNN_HYPER_PARAMS, 
                                                            cache_dir, seed==999))
            p.start()
            p.join()
        else:
            logging.info("Model for fold with seed %d found" % seed)
    
