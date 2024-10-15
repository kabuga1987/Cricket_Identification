from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D
from tensorflow.keras.layers import Input, GlobalMaxPooling2D,  MaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import os
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
from sklearn.metrics import precision_recall_curve

# ======================================================

class DataLoader:
    
    def __init__(self, path_train,  path_validation, samples_per_ID):
        self.path_train = path_train
        self.path_validation = path_validation
        self.samples_per_ID = samples_per_ID
        
    def slice_data_frame(self, path):
        """Select Segment, and ID columns from a data frame"""
        df = pd.read_csv(path)
        return df[["Segment", "ID"]]
        
    def sample_to_ID(self, df):
        """Map each sample to its ID given a data frame of samples and IDs"""
        return dict([(sample, ID) for sample, ID in df.to_records(index = False)])
    
    def ID_to_samples(self, sample_2_ID_dict):
        """Group samples per ID"""
        ID_2_samples = {}
        for sample, ID in sample_2_ID_dict.items():
            if ID not in ID_2_samples:ID_2_samples[ID] = [sample]
            else:ID_2_samples[ID].append(sample)
        return ID_2_samples
    
    def row_per_ID_matrix(self, ID_2_samples_dict):
        """Construct a matrix whose rows are grouped samples per ID.
        Padd rows with few samples by zeros to match the size of the ID with maximum samples"""
        
        lst_of_lst = list(ID_2_samples_dict.values())
        _max = max([len(lst) for lst in lst_of_lst])
        lst_of_all_imgs = np.zeros((len(lst_of_lst), _max),dtype='<U22')
        for i, lst in enumerate(lst_of_lst):
            padding_size = _max - len(lst)
            zeros = np.zeros(padding_size,)
            padded_with_zeros = np.hstack([lst,zeros])
            lst_of_all_imgs[i] = padded_with_zeros
        return lst_of_all_imgs
    
    def list_of_all_samples(self, ID_matx):
        """Construct a long sequence of samples by sequentially taking self.samples_per_ID.
        Remove the padded zeros to remain with actual samples"""
        
        L=[]
        N = ID_matx.shape[1]
        mults = np.arange(0, N, self.samples_per_ID)
        for i in mults:
            lst = ID_matx[:,i:i+self.samples_per_ID].flatten().tolist()
            L   = np.hstack([L,lst])
        return L[L!='0.0']
    
    def sample_2_label_ID(self, lst_of_all_samples, ID_2_samples_dict):
        """Map each sample to a numerical label ID"""
        sample_2_label = dict([(a[j], i) for i, a in enumerate(ID_2_samples_dict.values()) for j in range(len(a))])
        return zip(*[(sample, sample_2_label[sample]) for sample in lst_of_all_samples])
    
    
    def execution(self, path):
        df = self.slice_data_frame(path)
        sample_2_ID = self.sample_to_ID(df)
        ID_2_samples = self.ID_to_samples(sample_2_ID)
        ID_matrix = self.row_per_ID_matrix(ID_2_samples)
        all_samples = self.list_of_all_samples(ID_matrix)
        samples, labels = self.sample_2_label_ID(all_samples,ID_2_samples)
        return np.array(samples), np.array(labels)
    
    def train_validation_samples_labels(self):
        
        trn_samples, trn_labels = self.execution(self.path_train)
        val_samples, val_labels = self.execution(self.path_validation)
        
        return trn_samples, trn_labels, val_samples, val_labels
        


class Preprocessing():
    
    def __init__(self, path_2_imarray, meanvar):
        self.path_2_imarray = path_2_imarray
        self.meanvar = meanvar
    
    
    def load_imarray(self, img):
        img = np.load(self.path_2_imarray + img + ".npy")
        #img = self.mel_spec_2_log_spec(img)
        return np.expand_dims(img, axis = 2)

    def mel_spec_2_log_spec(self, mel_spectrogram):
        return librosa.power_to_db(mel_spectrogram, ref=np.min)
        
    
    def zeromean_unitvariance_normalizer(self,img):
        imarray = self.load_imarray(img)
        imarray -= np.mean(imarray, keepdims = True)
        imarray /= np.std(imarray, keepdims = True) + K.epsilon()
        return imarray
    
    def min_max_normalizer(self, img, _min = 0, _max = 1):
        imarray = self.load_imarray(img)
        norm_imarray = (imarray - imarray.min()) / (imarray.max() - imarray.min())
        norm_imarray = norm_imarray * (_max - _min) + _min
        return np.expand_dims(norm_imarray, axis = 2)
    
    def preprocess(self, img):
        return self.zeromean_unitvariance_normalizer(img) if self.meanvar else self.min_max_normalizer(img)
    
    
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
        ImsL  = np.zeros((size,), dtype = K.floatx())
        for i in range(size):
            ImsA[i,:,:,:] = super().preprocess(self.Ims[start + i])
            ImsL[i]     = self.Labels[start+i]
            
        return ImsA, ImsL
    
    def __len__(self):
        return (len(self.Ims) + self.batch_size - 1)//self.batch_size




class TestGe(Sequence, Preprocessing):

    
    def __init__(self, TeIs, shp, batch_size, p_arrays, meanvar = True):
        self.TeIs = TeIs
        self.shp = shp
        self.batch_size = batch_size
        self.p_arrays = p_arrays
        self.meanvar = meanvar
        super().__init__(self.p_arrays, self.meanvar)
          
    def __getitem__(self, index):
        start = self.batch_size*index
        size  = min(len(self.TeIs) - start, self.batch_size)
        ImsA  = np.zeros((size,) + self.shp, dtype=K.floatx())
        for i in range(size):
            ImsA[i,:,:,:] = super().preprocess(self.TeIs[start + i])
        return ImsA
    
    def __len__(self):
        return (len(self.TeIs) + self.batch_size - 1)//self.batch_size




class Evaluation:
    
    def compute_dist(self, Lembs, Rembs):
        return np.linalg.norm(Lembs - Rembs, axis = 1)
    
    def calc_acc(self, pairwise_distances, Y, thro):
        pres = np.where(pairwise_distances <= np.abs(thro),1,0)
        return np.where(Y == pres)[0].shape[0] / Y.shape[0]
         
    def prec_recal_thr(self, pairwise_distances, Y):
        return precision_recall_curve(Y, - pairwise_distances)
    
    def optimized_metrics(self, prec, recal, thr):
        f1s =  2 * prec * recal / (prec + recal)
        idx = np.argmax(f1s)
        thrOpt = round(np.abs(thr)[idx], ndigits = 3)
        precOpt = round(prec[idx], ndigits = 3)
        recalOpt = round(recal[idx],ndigits = 3)
        f1sOpt = round(f1s[idx],ndigits = 3)
        return precOpt,recalOpt,f1sOpt,thrOpt
    
    def plot_distances(self, pairwise_distances, Y, thro, R, P, F, A,night):
        id0, id1 = np.where(Y == 0), np.where(Y ==1 )
        m, u = pairwise_distances[id1], pairwise_distances[id0]
        r1 = np.arange(u.shape[0])
        r2 = np.arange(u.shape[0], u.shape[0] + m.shape[0])
        #plt.style.use("dark_background")
        plt.figure(figsize = (12,6))
        plt.scatter(r1, u, color = 'r', label = "Non-matches")
        plt.scatter(r2, m, color = 'g', label = "Matches")
        plt.title(f"Night = {night}, Recall = {R}, Precision = {P}, F1 = {F}, Accuracy = {A}")
        plt.axhline(y = abs(thro), color = 'b', linestyle = '-',label="Threshold")
        plt.xlabel("Number of pairs")
        plt.ylabel("Distance between image pairs")
        plt.legend()
        plt.savefig("figures/"+str(night)+".pdf")
        
    def evaluation_metrics(self, Lembs, Rembs, Y,night):
        pairwise_distances = self.compute_dist(Lembs, Rembs)
        prec, recal, thr = self.prec_recal_thr(pairwise_distances, Y)
        preco, recalo, f1so, thro = self.optimized_metrics(prec, recal, thr)
        acco = self.calc_acc(pairwise_distances, Y, thro)
        acco = round(acco, ndigits=4)
        self.plot_distances(pairwise_distances, Y, thro, recalo, preco, f1so, acco,night)
        preds = np.where(pairwise_distances <= thro,1,0)
        return preco, recalo, f1so, acco, thro, pairwise_distances, preds
    
    def mymetrics(self, embsL, embsR, Y, othr,night):
        pair_wise_dists = self.compute_dist(embsL, embsR)
        preds = np.where(pair_wise_dists <= np.abs(othr),1,0)
        TP = np.where((Y == 1) & (preds == 1))[0].shape[0]
        TN = np.where((Y == 0) & (preds == 0))[0].shape[0]
        FP = np.where((Y == 0) & (preds == 1))[0].shape[0]
        FN = np.where((Y == 1) & (preds == 0))[0].shape[0]
        Recall = round(TP/(TP+FN),ndigits = 4)
        Precision = round(TP/(TP+FP), ndigits = 4)
        F1s = round(2 * Precision * Recall / (Precision + Recall), ndigits=4)
        Accuracy = round((TP+TN)/(TP+TN+FP+FN),ndigits = 4)
        self.plot_distances(pair_wise_dists, Y, othr, Recall, Precision, F1s, Accuracy,night)
        return Precision, Recall, F1s, Accuracy, pair_wise_dists, preds


class ModelArchitecture(object):
    
    def __init__(self, shp, l2 = 0.0, k1 = 1, k2 = 2, k3 = 3, k9 = 3, mid = 32):
        
        self.k1,self.k2,self.k3,self.k9,self.mid = k1,k2,k3,k9,mid
        self.regul = regularizers.l2(l2)
        self.kwargs = {'padding':'same', 'kernel_regularizer':self.regul}
        self.s2 = self.k2
        self.shp = shp
         
    def EmbeddingModel(self):
        Ims = Input(shape = self.shp)
        x = self.FirstBlock(Ims,64)
        x = self.ConvBlockSubblock(x,128,64)
        x = self.ConvBlockSubblock(x,256,64)
        x = self.ConvBlockSubblock(x,384,96)
        x = self.ConvBlockSubblock(x,512,128)
        Embs = GlobalMaxPooling2D()(x)
        Embmodel = Model(Ims,Embs)
        return Embmodel
    
    
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
        
        model = self.architecture.EmbeddingModel()
        model.compile(loss=tfa.losses.TripletSemiHardLoss(), optimizer=Adam(Lr))# TripletSemiHardLoss,TripletHardLoss
        Kcallback=[ModelCheckpoint(pw,
                                   monitor = 'val_loss',
                                   save_best_only=True,
                                   save_weights_only = True),
        ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr= lr, verbose=0)]
    
    
        print()
        print("==================================================")
        print()
        print("Training model has started")
        print()
        print()
        print("++++++++++++++++++++++++++++++++++++++++++++++++++")
        print()
#         print("Training images = ", len(Ts))
#         print("Validation images = ", len(Vs))
#         print("Number of epochs = ", Epochs)
#         print("Learning rate upper limit = ", Lr)
#         print("Learning rate lower limit = ", lr)
#         print()
#         print("++++++++++++++++++++++++++++++++++++++++++++++++++")

        
        history = model.fit(self.Tgenerator,
                    validation_data = self.Vgenerator,
                    epochs = Epochs,
                    callbacks = Kcallback,
                    max_queue_size = 12,
                    workers = 6,
                           verbose=2)
        
        print()
        print("++++++++++++++++++++++++++++++++++++++++++++++++++")
        model.load_weights(pw)
        print()
        #print("Val loss = ", model.evaluate(self.Vgenerator, verbose=0))
        print()
        print()
        print("+++++++++++++++++++++++++++++++++++++")
        print()
        
        return model


