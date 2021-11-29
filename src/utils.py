from tensorflow.keras.callbacks import Callback
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)


class PlotL1RUL(tf.keras.callbacks.Callback):
    
    def __init__(self, X, Y, W):
        self.X = X
        self.Y = Y
        self.W = W
        
        self.__fig, axs = plt.subplots(2, 2, figsize=(20, 10))
        self._axs = [ax for aux in axs for ax in aux ]
        
        self.__plotted = False
        self.__count = 0

    def on_train_batch_end(self, batch, logs=None):
        if self.__count != 0:
            for ax in self._axs:
                ax.cla()
        
        self.__count += 1
        self.__fig.suptitle('4 random units from validation set')
        
        if (self.__count % 20) == 0:
            W = self.W

            predictions = defaultdict(lambda : {'y': [], 'p': []})
            for _id in list(self.X.keys()):
                batch = []
                y = []
                unit = self.X[_id]
                for c in range (0, unit.shape[0]-W, 5000):
                    batch.append(unit[c: c + W].T)
                    y.append(self.Y[_id][c + W])

                batch = np.array(batch)
                batch = batch.reshape(batch.shape + (1,))

                p = self.model.predict(batch, batch_size = 256, verbose=False)

                predictions[_id]['y'] = y
                predictions[_id]['p'] = p.reshape((p.shape[0],))

            for i,u in enumerate(predictions.keys()):
                self._axs[i].plot(predictions[u]['y'])
                self._axs[i].plot(predictions[u]['p'])
                self._axs[i].set_xticks([])

            self.__fig.savefig('train_progress.png', format='png')    



