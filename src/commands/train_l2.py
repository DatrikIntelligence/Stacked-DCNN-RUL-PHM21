#!/usr/bin/python
import sys
import os
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import argparse
from data import prepare_l2_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import pickle as pk
from models import *
import tensorflow as tf
import h5py
import gc
from scoring import *
import pandas as pd
import multiprocessing
from data import EmbeddingsGenerator, describe_features
from utils import PlotL2RUL
import logging
import random 

logging.basicConfig(level=logging.INFO)



FEATURES = ['alt', 'Mach', 'TRA', 'T2', 'T24', 'T30', 'T48', 'T50', 'P15', 'P2',
           'P21', 'P24', 'Ps30', 'P40', 'P50', 'Nf', 'Nc', 'Wf', 'Fc', 'hs']

           
def train(seed, W, batch_size, epochs, channels, model_config, input_embedding_size, step, cache_dir, net_summary=True): 
        logging.info("Starting training for fold with seed %d" % seed)
        
        # crate the model
        model = create_cnn_model((100,W,channels), **model_config)
        if net_summary:
            for l in str(model.summary()).split('\n'):
                logging.info(l)
        
        # read fold data sets and create generators
        try:
            
            logging.info("Loading data sets...")
            data = {}
            h5f = h5py.File(os.path.join(cache_dir, "train_test_l2_%d.h5" % seed), 'r')
            ids = h5f['train_ids'][:]
            y = {}
            for _id in ids:
                y[_id] = h5f['y_%d' % _id][:]
                data[_id] = h5f['x_%d' % _id][:]
                
            train_gen = EmbeddingsGenerator(data, y, seed, input_embedding_size=input_embedding_size)
 
            ids = h5f['test_ids'][:]
            y = {}
            data = {}
            for _id in ids:
                y[_id] = h5f['y_%d' % _id][:]
                data[_id] = h5f['x_%d' % _id][:]
                
            test_gen = EmbeddingsGenerator(data, y, seed, input_embedding_size=input_embedding_size)
            
            h5f.close()
            
            
        except Exception as ex:
            raise Exception('Error reading data', ex)
        
        train_gen.window = W
        train_gen.epoch_len_reducer = 2000
        train_gen.batch_size = batch_size
        train_gen.return_label = True
        train_gen.channels = channels
        train_gen.step = step
        train_gen.transpose_input = False
        
        test_gen.window = W
        test_gen.batch_size = 256
        test_gen.epoch_len_reducer = 400
        test_gen.return_label = True
        test_gen.channels = channels
        test_gen.step = step
        test_gen.transpose_input = False
        
        logging.info("Data sets already loaded...")
        

        # plot rul data
        ids = random.sample(list(test_gen._X.keys()), 4)
        X_rul = {_id:test_gen._X[_id] for _id in ids}
        Y_rul = {_id:test_gen._Y[_id] for _id in ids}
        
        # train
        es = tf.keras.callbacks.EarlyStopping(monitor='val_Score', patience=8)
        rlr = tf.keras.callbacks.ReduceLROnPlateau(patience=3)
        pr = PlotL2RUL(X_rul, Y_rul, W, step)
        history = model.fit(train_gen, validation_data=test_gen, 
                            epochs=epochs, verbose=2, 
                            callbacks=[es, rlr, pr])
        
        history_path = os.path.join(cache_dir, 'cnn_l2_history_%d.pk' % seed)
        pk.dump(history.history, open(history_path, 'wb'))
        model_path = os.path.join(cache_dir, 'cnn_l2_%d.h5' % seed)
        model.save(model_path)
            
        
CNN_HYPER_PARAMS = {
    'scorer': 1, 
    'block_size': 4.41106, 
    'conv_activation': 1.69354, 
    'dense_activation': 1.28829, 
    'dilation_rate': 1.839, 
    'dropout': 0.131517, 
    'fc1': 247, 
    'fc2': 105, 
    'kernel_size': 0.688211, 
    'l1': 0.000696, 
    'l2': 0.0000173, 
    'lr': 0.000553, 
    'nblocks': 4.02792, 
}


BATCH_SIZE = 31
W = 100
EPOCHS = 100
CHANNELS = 3
INPUT_EMB_SIZE = 162
STEP = 989
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train L2 model')
    args = parser.parse_args()
    cache_dir = prepare_l2_data()
    
    for seed in [999, 666, 128, 256, 394]:  
        model_path = os.path.join(cache_dir, 'cnn_l2_%d.h5' % seed)
        
        if not os.path.exists(model_path):
            queue = multiprocessing.Queue()
            p = multiprocessing.Process(target=train, args=(seed, W, BATCH_SIZE, EPOCHS, CHANNELS, CNN_HYPER_PARAMS, 
                                                            INPUT_EMB_SIZE, STEP, cache_dir, seed==999))
            p.start()
            p.join()
        else:
            logging.info("Model for fold with seed %d found" % seed)
    
