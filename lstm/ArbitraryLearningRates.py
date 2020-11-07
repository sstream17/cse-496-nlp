import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import *
import tensorflow.keras.backend as backend

# rates: list of arbitrary learning rates
# patience: number of epochs with no improvement to determine a plateau
# restore_best_weights: if True, the end weights will be the best weights found
#                       the best weights of the previous learning rate are NOT
#                       restored when resuming with the next learning rate
class LearningRateCallback(tf.keras.callbacks.Callback):
    def __init__(self, rates=[0.005, 0.001, 0.0005], patience=25, restore_best_weights = False, early_stopping=True):
        super(LearningRateCallback, self).__init__()
        self.rates = rates
        self.rate_index = 0
        self.patience = patience
        self.use_best = restore_best_weights # flag
        self.best_weights = None
        self.early_stopping = early_stopping
    
    # Initialize the starting values for training
    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        print("Beginning fit with learning rate " + str(self.rates[0]))
        self.best = np.Inf
    
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        # Check if the loss is still decreasing
        if np.less(current, self.best):
            # If it is we're not waiting to see if it improves
            self.best = current
            self.wait = 0
            
            # Must be the current best weights
            if self.use_best:
                self.best_weights = self.model.get_weights()
        else:
            # Increment the wait period and see if we've run out of patience
            self.wait += 1
            if self.wait >= self.patience:
                # If we still have rates grab the next one
                if self.rate_index < len(self.rates) - 1:
                    # Reset time spent waiting
                    self.wait = 0
                    
                    # Increment the rate index and set to that rate
                    self.rate_index += 1
                    backend.set_value(self.model.optimizer.lr, self.rates[self.rate_index])
                    print("Changing rate to " + str(self.rates[self.rate_index]) + " on epoch " + str(epoch))
                elif self.early_stopping:
                    # Ran out of rates to try so we'll stop
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    
                    # If asked for, use only the best weights
                    if self.use_best:
                        self.model.set_weights(self.best_weights)
                        print("Restoring the best weights")
                        
    def on_train_end(self,logs=None):
        if self.stopped_epoch > 0:
            print("Early stopping on epoch " + str(self.stopped_epoch))
    
    
    
    
    
    