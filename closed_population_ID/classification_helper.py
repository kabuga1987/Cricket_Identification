from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D
from tensorflow.keras.layers import Input, GlobalMaxPooling2D,  MaxPooling2D, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence, to_categorical
from sklearn.metrics import accuracy_score
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
    
    def __init__(self, shp, l2 = 0.0, k1 = 1, k2 = 2, k3 = 3, k9 = 7, mid = 32):
        
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
        x = Dense(3,activation = 'softmax')(x) # the number of classes is 47 for the full data
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

#===================================
class LabelEncoder:
    """
    Encode segment labels into one-hot vectors
    for training, validation, and testing.
    """

    def __init__(self, path_csv, path_samples):
        """
        Parameters
        ----------
        path_csv : str
            CSV file containing segment-to-ID mappings.

        path_samples : str
            Directory containing train/validation/test split CSV files.
        """
        self.path_csv = path_csv
        self.n1_trn = os.path.join(path_samples,"n1_train.csv")
        self.n1_val = os.path.join(path_samples,"n1_validation.csv")
        self.n1_tst = os.path.join(path_samples,"n1_test.csv")
        self.n2_tst = os.path.join(path_samples,"n2_test.csv")
        self.n3_tst = os.path.join(path_samples,"n3_test.csv")

    @staticmethod
    def slice_data_frame(path):
        """
        Read only Segment and ID columns.
        Parameters
        ----------
        path : str, CSV file path.
        Returns
        -------
        pd.DataFrame, Dataframe containing Segment and ID columns.
        """
        df = pd.read_csv(path)
        required_columns = ["Segment", "ID"]
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            raise ValueError(
                f"Missing columns: {missing_cols}"
            )

        return df[required_columns]

    @staticmethod
    def sample_to_ID(df):
        """
        Map each sample to its corresponding ID.
        Parameters
        ----------
        df : pd.DataFrame
        Returns
        -------
        dict, {sample: ID}
        """
        return dict(zip(df["Segment"], df["ID"]))

    @staticmethod
    def ID_to_samples(sample_2_ID_dict):
        """
        Group samples by ID.
        Parameters
        ----------
        sample_2_ID_dict : dict
        Returns
        -------
        dict, {ID: [samples]}
        """
        ID_2_samples = {}
        for sample, ID in (sample_2_ID_dict.items()):
            if ID not in ID_2_samples:
                ID_2_samples[ID] = []
            ID_2_samples[ID].append(sample)
        return ID_2_samples

    @staticmethod
    def get_samples(path):
        """
        Load sample names from CSV file.
        Parameters
        ----------
        path : str
        Returns
        -------
        np.ndarray, Array of segment names.
        """
        df = pd.read_csv(path)
        if "Segment" not in df.columns:
            raise ValueError(
                f"'Segment' column missing in {path}"
            )
        return df["Segment"].values

    def hot_encode_labels(self, lst_samples, ID_2_samples_dict):
        """
        Convert sample IDs to one-hot encoded labels.
        Parameters
        ----------
        lst_samples : array-like
            Segment names.
        ID_2_samples_dict : dict
            {ID: [samples]}
        Returns
        -------
        tuple
            samples_array, labels_array
        """
        # Assign numeric label to each ID
        sample_to_numeric_ID = {}
        for numeric_ID, samples in enumerate(ID_2_samples_dict.values()):
            for sample in samples:
                sample_to_numeric_ID[sample] = numeric_ID
        # Create one-hot labels
        all_samples = list(sample_to_numeric_ID.keys())

        numeric_labels = [
            sample_to_numeric_ID[sample]
            for sample in all_samples
        ]

        one_hot_labels = to_categorical(numeric_labels)

        sample_to_onehot = {
            all_samples[i]: one_hot_labels[i]
            for i in range(len(all_samples))
        }

        # Keep only requested samples
        valid_samples = []
        valid_labels = []
        for sample in lst_samples:
            if sample not in sample_to_onehot:
                continue
            valid_samples.append(sample)
            valid_labels.append(sample_to_onehot[sample])

        return (np.array(valid_samples), np.array(valid_labels))

    @staticmethod
    def shuffle_data(samples, labels):
        """
        Shuffle samples and labels jointly.
        Parameters
        ----------
        samples : np.ndarray
        labels : np.ndarray
        Returns
        -------
        tuple, (shuffled_samples, shuffled_labels)
        """
        indices = np.arange(len(samples))
        np.random.shuffle(indices)
        return (samples[indices],labels[indices])

    def HotEncodeLabels(self):
        """
        Generate one-hot encoded datasets.
        Returns
        -------
        tuple: (Train, validation, test)
        """
        df = self.slice_data_frame(self.path_csv)
        sample_2_ID = self.sample_to_ID(df)
        ID_2_samples = self.ID_to_samples(sample_2_ID)
        # Load split samples
        n1_trn_s = self.get_samples(self.n1_trn)
        n1_val_s = self.get_samples(self.n1_val)
        n1_tst_s = self.get_samples(self.n1_tst)
        n2_tst_s = self.get_samples(self.n2_tst)
        n3_tst_s = self.get_samples(self.n3_tst)

        # Encode labels
        n1ts, n1ty = self.hot_encode_labels(n1_trn_s, ID_2_samples)
        n1vs, n1vy = self.hot_encode_labels(n1_val_s, ID_2_samples)
        n1es, n1ey = self.hot_encode_labels(n1_tst_s, ID_2_samples)
        n2es, n2ey = self.hot_encode_labels(n2_tst_s, ID_2_samples)
        n3es, n3ey = self.hot_encode_labels(n3_tst_s, ID_2_samples)

        # Shuffle datasets
        n1ts, n1ty = self.shuffle_data(n1ts, n1ty)
        n1vs, n1vy = self.shuffle_data(n1vs, n1vy)
        n1es, n1ey = self.shuffle_data(n1es, n1ey)
        n2es, n2ey = self.shuffle_data(n2es, n2ey)
        n3es, n3ey = self.shuffle_data(n3es, n3ey)
        return (
            n1ts, n1ty,
            n1vs, n1vy,
            n1es, n1ey,
            n2es, n2ey,
            n3es, n3ey
        )

class Predictions:
    """
    Generate predictions, compute accuracies,
    and save prediction results.
    """

    def __init__(
        self,
        model,
        v1x,
        v1y,
        n1x,
        n1y,
        n2x,
        n2y,
        n3x,
        n3y,
        vgen,
        gen1,
        gen2,
        gen3,
        p_preds
    ):
        """
        Parameters
        ----------
        model : keras.Model
            Trained classification model.
        v1x, n1x, n2x, n3x : np.ndarray
            Sample names.

        v1y, n1y, n2y, n3y : np.ndarray
            One-hot encoded labels.

        vgen, gen1, gen2, gen3 :
            Data generators corresponding to each dataset.

        p_preds : str
            Directory where prediction CSV files
            will be saved.
        """

        self.model = model
        self.v1x = v1x
        self.v1y = v1y
        self.n1x = n1x
        self.n1y = n1y
        self.n2x = n2x
        self.n2y = n2y
        self.n3x = n3x
        self.n3y = n3y
        self.vgen = vgen
        self.gen1 = gen1
        self.gen2 = gen2
        self.gen3 = gen3
        self.p_preds = p_preds
        os.makedirs(
            self.p_preds,
            exist_ok=True
        )

    def predict(
        self,
        samples,
        hotlabels,
        generator,
        dataset_name,
        verbose=0
    ):
        """
        Predict labels and compute accuracy.
        Parameters
        ----------
        samples: np.ndarray, Sample names.
        hotlabels: np.ndarray, One-hot encoded labels.
        generator: Data generator used for prediction.
        dataset_name: str, Name used when saving prediction CSV.
        verbose : int, Verbosity level.
        Returns
        -------
        float, Classification accuracy.
        """

        # Model predictions
        predictions = self.model.predict(generator, verbose=verbose)

        # Convert probabilities to class indices
        observed = np.argmax(hotlabels, axis=1)
        predicted = np.argmax(predictions, axis=1)
        # Accuracy
        accuracy = accuracy_score(observed, predicted)
        print(f"Accuracy for {dataset_name}: "f"{accuracy:.4f}")
        # Save predictions
        df = pd.DataFrame({"Sample": samples,"Observed": observed,"Predicted": predicted})
        output_path = os.path.join(self.p_preds, f"{dataset_name}.csv")
        df.to_csv(output_path,index=False)
        return accuracy

    def execution(self):
        """
        Run predictions on validation and test sets.
        Returns
        -------
        pd.DataFrame, Accuracy summary dataframe.
        """
        # Validation
        av = self.predict(
            samples=self.v1x,
            hotlabels=self.v1y,
            generator=self.vgen,
            dataset_name="one_n1_val"
        )
        print()
        # Night 1 test
        a1 = self.predict(
            samples=self.n1x,
            hotlabels=self.n1y,
            generator=self.gen1,
            dataset_name="one_n1_test"
        )
        print()
        # Night 2 test
        a2 = self.predict(
            samples=self.n2x,
            hotlabels=self.n2y,
            generator=self.gen2,
            dataset_name="one_n2_test"
        )

        print()

        # Night 3 test
        a3 = self.predict(
            samples=self.n3x,
            hotlabels=self.n3y,
            generator=self.gen3,
            dataset_name="one_n3_test"
        )
        # Accuracy summary
        metrics_df = pd.DataFrame({
            "Dataset": [
                "night1_validation",
                "night1_test",
                "night2_test",
                "night3_test"
            ],
            "Accuracy": [
                round(av, 3),
                round(a1, 3),
                round(a2, 3),
                round(a3, 3)
            ]
        })

        metrics_output = os.path.join(self.p_preds,"one_class_metrics.csv")
        metrics_df.to_csv(metrics_output, index=False)
        return metrics_df


    