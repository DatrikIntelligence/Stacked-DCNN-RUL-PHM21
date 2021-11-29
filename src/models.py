from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Activation, MaxPooling2D, Reshape, Input, Dropout
from tensorflow.keras.layers import BatchNormalization, Lambda, Conv2DTranspose, Add
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras import constraints
import tensorflow.keras.backend as K
import numpy as np
import inspect
from scoring import *
from tensorflow.keras.layers.experimental.preprocessing import Resizing
    
activations = ['relu', tf.keras.layers.LeakyReLU(alpha=0.1), 'tanh']
kernels = [(3,3), (10, 1), (10, 5)]      
scorers = [None, 'mean_squared_error']
def create_cnn_model(input_shape, block_size=2, nblocks=2, l1=1e-5, l2=1e-4, 
                     kernel_size=0, dropout=0.5, lr=1e-3, fc1=256, fc2=128,
                     conv_activation=2, dense_activation=2, dilation_rate=1,
                     batch_normalization=1, scorer=1):
    block_size = int(round(block_size))
    nblocks = int(round(nblocks))
    scorer = scorers[int(round(scorer))]
    fc1 = int(round(fc1))
    fc2 = int(round(fc2))
    dilation_rate = int(round(dilation_rate))
    conv_activation = activations[int(round(conv_activation))]
    dense_activation = activations[int(round(dense_activation))]
    kernel_size = kernels[int(round(kernel_size))]
    batch_normalization = True if batch_normalization == 1 else False
    
    input_tensor = Input(input_shape)
    x = input_tensor
    for i, n_cnn in enumerate([block_size] * nblocks):
        for j in range(n_cnn):
            x = Conv2D(32*2**min(i, 2), kernel_size=kernel_size, padding='same', 
                       kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                       kernel_initializer='he_uniform',
                       dilation_rate=dilation_rate,
                       name='Conv%d_Block%d' % (j, i) )(x)
            if batch_normalization:
                x = BatchNormalization(name='BN%d_Block%d' % (j, i))(x)
            x = Activation(conv_activation, name='A%d_Block%d' % (j, i))(x)
        x = MaxPooling2D(2, name='MP%d_Block%d' % (j, i))(x)
        if dropout > 0:
            x = Dropout(dropout, name='DO%d_Block%d' % (j, i))(x)

    x = Flatten()(x)
    
    # FNN
    x = Dense(fc1, name='Fc1', 
             kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2))(x)
    x = Activation(dense_activation, name='Act_Fc1', )(x)
    if dropout > 0:
        x = Dropout(dropout, name='DO_Fc1')(x)
    x = Dense(fc2, name='Fc2', 
             kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2))(x)
    x = Activation(dense_activation, name='Act_Fc2')(x)
    if dropout > 0:
        x = Dropout(dropout, name='DO_Fc2')(x)
    x = Dense(1, activation='relu', name='predictions')(x) 
    model = Model(inputs=input_tensor, outputs=x)
    model.compile(loss=scorer, optimizer=Adam(lr=lr), 
                  metrics=[NASAScore(), PHM21Score(), tf.keras.metrics.MeanAbsoluteError(name="MAE")])
    
    return model

