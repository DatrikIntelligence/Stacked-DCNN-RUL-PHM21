import os
import pathlib
import random
import numpy as np 
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from collections import Counter
from pyts.image import GramianAngularField
from pandas import DataFrame
import pandas as pd
import h5py
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle as pk
import gc
import logging

logging.basicConfig(level=logging.INFO)

# deactive panda warning
pd.options.mode.chained_assignment = None

class DataGenerator(Sequence):
    """
    """
    def __init__(self, X, attributes, window_size=10, batch_size=32, 
                 noise_level=0, epoch_len_reducer=100, add_extra_channel=False,
                 return_label=True):
        self.batch_size = batch_size
        self.return_label = return_label
        self.window_size = window_size
        self.noise_level = noise_level
        self.attributes = attributes
        self.epoch_len_reducer = epoch_len_reducer
        self._X = {}
        self._Y = {}
        self._ids = X.id.unique()
        self.add_extra_channel = add_extra_channel
        for _id in self._ids:
            self._X[_id] = X.loc[(X.id==_id), self.attributes].values
            self._Y[_id] = X.loc[(X.id==_id), 'Y'].values
        self.__len = int((X.groupby('id').size() - self.window_size).sum() / 
                        self.batch_size)
        del X


    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(self.__len / self.epoch_len_reducer)
    
    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        X = self._X
        _X = []
        _y = []
        for _ in range(self.batch_size):
            sid = random.choice(self._ids)
            unit = self._X[sid]
            nrows = unit.shape[0]
            cut = random.randint(0, nrows - self.window_size)
            s = unit[cut: cut + self.window_size].T
            y =self._Y[sid][cut + self.window_size-1]
            _X.append(s)
            _y.append(y)

        
        _X = np.array(_X)
        if self.add_extra_channel:
            _X = _X.reshape(_X.shape + (1,))
            
        if self.noise_level > 0:
            noise_level = self.noise_level
            noise = np.random.normal(-noise_level, noise_level, _X.shape)
            _X = _X + noise
            _X = (_X - _X.min()) / (_X.max() - _X.min())
       
        if self.return_label:
            return _X, np.array(_y).reshape((self.batch_size, 1))
        else:
            return _X, _X
        
    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        pass


class EmbeddingsGenerator(Sequence):
    """
    """
    def __init__(self, X, Y, seed, batch_size=32, 
                 epoch_len_reducer=100, window_size = 100, 
                 channels=1, step=1000, input_embedding_size=1000,
                 transpose_input=False):
        self.batch_size = batch_size
        
        self.epoch_len_reducer = epoch_len_reducer
        self._ids = list(X.keys())
        self._Y = Y
        self._X = X
        self.seed = seed
        self.__len = sum(len(Y[_id]) for _id in Y.keys())
        self.channels = channels
        self.window = window_size
        self.step = step
        self.input_embedding_size = input_embedding_size
        self.transpose_input = transpose_input
        

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(self.__len / self.epoch_len_reducer / self.batch_size)
    
    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        jump = self.step
        latent_size = 100
        ids = np.random.choice(self._ids, size=self.batch_size, replace=True)
        ids = Counter(sorted(ids))
        channels = self.channels if self.channels else 1
        encoders = np.zeros((self.batch_size*channels*self.window, latent_size))
        y = np.zeros((self.batch_size,))
        ei = 0
        si = 0
        for _id, n in ids.items():
            nrows = len(self._Y[_id])

            data = self._X[_id]
            
            for _ in range(n):
                cut = random.randint(0, nrows - self.window * jump * channels - self.input_embedding_size)
                
                for j in range(self.window * channels):
                    c = cut + (jump * j)
                    encoders[ei, :] = data[c].T
                    
                    ei += 1
            
                y[si] = self._Y[_id][c + self.input_embedding_size]
                si += 1  
       
        _X = np.zeros((self.batch_size, self.window, latent_size, channels))
        nz = self.window * channels
        for i in range(self.batch_size): 
            _X[i] = encoders[i*nz:nz*(i+1),:].T.reshape(self.window, latent_size, channels)
            
        _X = _X.astype(np.float32)
        _y = np.array(y).reshape((self.batch_size, 1)).astype(np.float32)
        
        if not self.channels:
            _X = np.squeeze(_X, axis=-1)
        
        if self.transpose_input:
            _X = _X.transpose((0,2,1))        

        return _X, _y
    

        
    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        pass    

def load_phm21_data_file(filename):
    # Load data
    with h5py.File(filename, 'r') as hdf:
            # Development set
            W_dev = np.array(hdf.get('W_dev'), dtype=np.float32)             # W
            X_s_dev = np.array(hdf.get('X_s_dev'), dtype=np.float32)         # X_s
            Y_dev = np.array(hdf.get('Y_dev'), dtype=np.float32)             # RUL  
            A_dev = np.array(hdf.get('A_dev'), dtype=np.float32)             # Auxiliary

            # Test set
            W_test = np.array(hdf.get('W_test'), dtype=np.float32)           # W
            X_s_test = np.array(hdf.get('X_s_test'), dtype=np.float32)       # X_s
            Y_test = np.array(hdf.get('Y_test'), dtype=np.float32)           # RUL  
            A_test = np.array(hdf.get('A_test'), dtype=np.float32)           # Auxiliary

            # Varnams
            W_var = np.array(hdf.get('W_var'))
            X_s_var = np.array(hdf.get('X_s_var'))  
            A_var = np.array(hdf.get('A_var'))

            # from np.array to list dtype U4/U5
            W_var = list(np.array(W_var, dtype='U20'))
            X_s_var = list(np.array(X_s_var, dtype='U20'))  
            A_var = list(np.array(A_var, dtype='U20'))

    W = np.concatenate((W_dev, W_test), axis=0)  
    X_s = np.concatenate((X_s_dev, X_s_test), axis=0)
    Y = np.concatenate((Y_dev, Y_test), axis=0)
    A = np.concatenate((A_dev, A_test), axis=0) 

    X = np.concatenate((W, X_s, A, Y), axis=1)
    X = DataFrame(data=X, columns=W_var+X_s_var+A_var+['Y'])
    
    return X

def normalize_data(X_train, X_test, features):
    
    #scaler = MinMaxScaler()
    scaler = StandardScaler()
    values = scaler.fit_transform(X_train[features].values).round(3)
        
    for i,feature in enumerate(features):
        X_train[feature] = values[:, i]

    values = scaler.transform(X_test[features].values).round(3)
    
    for i,feature in enumerate(features):
        X_test[feature] = values[:, i]

    return X_train, X_test, scaler

    
def train_test_split(X, train_ratio=0.7, random_seed=999):
    random.seed(random_seed)
    ids = set(X.id.unique())
    train_ids = random.sample(ids, int(len(ids) * train_ratio))
    test_ids =  list(ids - set(train_ids))
    logging.info("Train ids:" + str(train_ids))
    logging.info("Test ids:" + str(test_ids))
    return X[X.id.isin(train_ids)].copy(), X[X.id.isin(test_ids)].copy()

def exists_all_files(filename_format, cache_dir):
    return all(os.path.exists(os.path.join(cache_dir, filename_format % seed)) 
               for seed in [999, 666, 128, 256, 394])

def describe_features(X, features):
    d = str(X[features].describe().T[['min', 'max', 'mean', 'std']]).split('\n')
    for l in d:
        logging.info(l)

def prepare_l1_data():
    # download the dataset
    logging.info("Checking if main dataset was downloaded and download in other case.")
    data_path = tf.keras.utils.get_file(
        fname='phm21_datset.zip',
        origin='https://ti.arc.nasa.gov/m/project/prognostic-repository/data_set.zip',
        cache_subdir='datasets', extract=False, 
    )
    
    if not os.path.exists(os.path.join(os.path.join(pathlib.Path(data_path).parent, 'data_set'))):
        data_path = tf.keras.utils.get_file(
            fname='phm21_datset.zip',
            origin='https://ti.arc.nasa.gov/m/project/prognostic-repository/data_set.zip',
            cache_subdir='datasets', extract=True, 
        )
        
    train_filename_format = 'train_l1_%d.h5'
    test_filename_format = 'test_l1_%d.h5'
    scaler_filename_format = 'scaler_%d.pk' 
    cache_dir = os.path.join(pathlib.Path(data_path).parent, 'data_set')

    if (not exists_all_files(train_filename_format, cache_dir) or 
        not exists_all_files(test_filename_format, cache_dir) or 
        not exists_all_files(scaler_filename_format, cache_dir)):
        
        # read data files
        files =  ['N-CMAPSS_DS01-005.h5', 'N-CMAPSS_DS03-012.h5', 'N-CMAPSS_DS04.h5', 'N-CMAPSS_DS05.h5',
                  'N-CMAPSS_DS06.h5', 'N-CMAPSS_DS07.h5', 'N-CMAPSS_DS08a-009.h5', 'N-CMAPSS_DS08c-008.h5']


        logging.info('Reading file:' + files[0])    
        X = load_phm21_data_file(os.path.join(cache_dir, files[0]))
        X['id'] = X.unit
        #X['file'] = files[0].replace('.h5', '')
        for file in files[1:]:
            logging.info('Reading file:' + file)   
            aux = load_phm21_data_file(os.path.join(cache_dir,  file))
            aux['id'] = aux.unit + X.id.max()
            #X['file'] = file.replace('.h5', '')
            X = pd.concat((X, aux), axis=0)

            del aux
            gc.collect()

        # memory needed: 12GB
        # create the cross-validation sets

        # feature to normalize
        features= ['alt', 'Mach', 'TRA', 'T2', 'T24', 'T30', 'T48', 'T50', 'P15', 'P2',
                   'P21', 'P24', 'Ps30', 'P40', 'P50', 'Nf', 'Nc', 'Wf', 'Fc', 'hs']

        for seed in [999, 666, 128, 256, 394]:
            train_set_file_path = os.path.join(cache_dir, 'train_l1_%d.h5' % seed)
            test_set_file_path = os.path.join(cache_dir, 'test_l1_%d.h5' % seed)
            scaler_file_path = os.path.join(cache_dir, 'scaler_%d.pk' % seed)
            if (not os.path.exists(train_set_file_path) or
                not os.path.exists(test_set_file_path)  or
                not os.path.exists(scaler_file_path)) :
                
                logging.info("Creating train and test set for cv fold with random seed %d" % seed)

                # split
                X_train, X_test = train_test_split(X, random_seed=seed)
                # memory needed: 14GB

                # data normalization
                X_train, X_test, scaler = normalize_data(X_train, X_test, features)

                # store
                pk.dump(scaler, open(scaler_file_path, 'wb'))
                X_train.to_hdf(train_set_file_path, key='phm21')
                X_test.to_hdf(test_set_file_path, key='phm21')

                del X_train
                del X_test
                #del train_gen
                #del text_gen
                gc.collect()
            
    return cache_dir
    