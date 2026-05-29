from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D
from tensorflow.keras.layers import Input, GlobalMaxPooling2D,  MaxPooling2D, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence, to_categorical
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import os
import matplotlib.pyplot as plt
import librosa

#===================================================

class Preprocessing:
    
    def __init__(self, path_2_imarray, meanvar):
        self.path_2_imarray = path_2_imarray
        self.meanvar = meanvar
    
    
    def load_imarray(self, img):
        img = np.load(self.path_2_imarray + img+".npy")
        #img = self.mel_spec_2_log_spec(img)
        return img

    def mel_spec_2_log_spec(self, mel_spectrogram):
        return librosa.core.power_to_db(mel_spectrogram)
        
    
    def zeromean_unitvariance_normalizer(self,img):
        imarray = self.load_imarray(img)
        imarray -= np.mean(imarray, keepdims = True)
        imarray /= np.std(imarray, keepdims = True) + K.epsilon()
        return np.expand_dims(imarray, axis = 2)
    
    def min_max_normalizer(self, img, _min = 0, _max = 1):
        imarray = self.load_imarray(img)
        norm_imarray = (imarray - imarray.min()) / (imarray.max() - imarray.min())
        norm_imarray = norm_imarray * (_max - _min) + _min
        return np.expand_dims(norm_imarray, axis = 2)
    
    
         
    
    def preprocess(self, img):
        return self.zeromean_unitvariance_normalizer(img) if self.meanvar else self.min_max_normalizer(img)
        
    
    
    def spec_augment(self, original_melspec, freq_masking_max_percentage = 0.15, time_masking_max_percentage = 0.15):
        
        # from https://towardsdatascience.com/data-augmentation-techniques-for-audio-data-in-python-15505483c63c

        augmented_melspec = original_melspec.copy()
        all_frames_num, all_freqs_num = augmented_melspec.shape

        # Frequency masking
        freq_percentage = random.uniform(0.0, freq_masking_max_percentage)
        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = int(np.random.uniform(low = 0.0, high = (all_freqs_num - num_freqs_to_mask)))

        augmented_melspec[:, f0:(f0 + num_freqs_to_mask)] = 0

        # Time masking
        time_percentage = random.uniform(0.0, time_masking_max_percentage)
        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = int(np.random.uniform(low = 0.0, high = (all_frames_num - num_frames_to_mask)))

        augmented_melspec[t0:(t0 + num_frames_to_mask), :] = 0

        return augmented_melspec


class Generator(Sequence, Preprocessing):

    
    def __init__(self, Ims, Labels, shp, batch_size, p_arrays, meanvar):
        
        self.Ims       = Ims
        self.Labels    = Labels
        self.shp = shp
        self.batch_size = batch_size
        self.p_arrays = p_arrays
        self.meanvar = meanvar
        
        super().__init__(self.p_arrays, self. meanvar)
          
    def __getitem__(self, index):
        start = self.batch_size*index
        size  = min(len(self.Ims) - start, self.batch_size)
        ImsA  = np.zeros((size,) + self.shp, dtype=K.floatx())
        ImsL  = np.zeros((size,) + (self.Labels.shape[1],) , dtype = K.floatx())
        for i in range(size):
            ImsA[i,:,:,:] = super().preprocess(self.Ims[start + i])
            ImsL[i]     = self.Labels[start+i]
            
        return ImsA, ImsL
    
    def __len__(self):
        return (len(self.Ims) + self.batch_size - 1)//self.batch_size
    


    
class ModelArchitecture(object):
    
    def __init__(self, shp, l2 = 0.0, k1 = 1, k2 = 2, k3 = 3, k9 = 9, mid = 32):
        
        self.k1,self.k2,self.k3,self.k9,self.mid = k1,k2,k3,k9,mid
        self.regul = regularizers.l2(l2)
        self.kwargs = {'padding':'same', 'kernel_regularizer':self.regul}
        self.s2 = self.k2
        self.shp = shp
         
    def ClassifyIDModel(self):
        Ims = Input(shape = self.shp)
        x = self.FirstBlock(Ims,64)
        x = self.ConvBlockSubblock(x,128,64)
        x = self.ConvBlockSubblock(x,256,64)
        x = self.ConvBlockSubblock(x,384,96)
        x = self.ConvBlockSubblock(x,512,128)
        x = GlobalMaxPooling2D()(x)
        x = Dense(256,activation='relu')(x)
        x = Dense(128,activation='relu')(x)
        x = Dense(3,activation = 'softmax')(x)
        return Model(Ims, x, name = "ID_classification_model")
    
    
    def Subblock(self,x, convF):
        x = BatchNormalization()(x)
        y = x
        y = Conv2D(convF, (self.k1, self.k1), activation='relu', **self.kwargs)(y) #reduce the nr of feature to filter
        y = BatchNormalization()(y)
        y = Conv2D(convF, (self.k3, self.k3), activation='relu', **self.kwargs)(y) # extend the feature field
        y = BatchNormalization()(y)
        y = Conv2D(K.int_shape(x)[-1], (self.k1, self.k1), **self.kwargs)(y)# restore the nr of original features
        y = Add()([x,y]) # add a skip connection
        y = Activation('relu')(y)
        return y
    
    
    def FirstBlock(self,x,convF):
        x   = Conv2D(convF, (self.k9,self.k9), strides=self.s2, activation='relu', **self.kwargs)(x)
        x   = MaxPooling2D((self.k2, self.k2),padding ="same",  strides=(self.s2, self.s2))(x) 
        for _ in range(2):
            x = BatchNormalization()(x)
            x = Conv2D(convF, (self.k3,self.k3), activation='relu', **self.kwargs)(x)
        return x
    
    
    def ConvBlockSubblock(self,x,convF,subbF):
        x = MaxPooling2D((self.k2, self.k2),padding ="same", strides=(self.s2, self.s2))(x) 
        x = BatchNormalization()(x)
        x = Conv2D(convF, (self.k1,self.k1), activation='relu', **self.kwargs)(x)
        for _ in range(4): x = self.Subblock(x, subbF)
        return x
    
          
        
class Training:
    
    def __init__(self):
    
        self.architecture = None
        self.Tgenerator = None
        self.Vgenerator = None
    
        
        
    def train(self, Lr, lr, pw, Epochs):
        
        model = self.architecture.ClassifyIDModel()
        
        model.compile(loss = "categorical_crossentropy", metrics = ["accuracy"], optimizer = Adam(Lr))
        
        Kcallback=[ModelCheckpoint(pw,
                                   monitor = 'val_loss',
                                   save_best_only=True,
                                   save_weights_only = True),
        ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr= lr, verbose=1)]
    
    
        print()
        print("==================================================")
        print()
        print("Training model has started")
        print()
        print()
        print("++++++++++++++++++++++++++++++++++++++++++++++++++")
        
        
        history = model.fit(self.Tgenerator,
                    validation_data = self.Vgenerator,
                    epochs = Epochs,
                    callbacks = Kcallback,
                    max_queue_size = 12,
                    workers = 6,
                           verbose=1)
        
        print()
        print("++++++++++++++++++++++++++++++++++++++++++++++++++")
        print()
        model.load_weights(pw)

        
        return model
    