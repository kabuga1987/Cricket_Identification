# from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D
# from tensorflow.keras.layers import Input, GlobalMaxPooling2D,  MaxPooling2D, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    BatchNormalization,
    Dropout
)
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import os
import matplotlib.pyplot as plt
import librosa

#===================================================

class Preprocessing:
    
    def __init__(self, path_2_imarray):
        self.path_2_imarray = path_2_imarray
    
    
    def load_imarray(self, img):
        img = np.load(self.path_2_imarray + img+".npy")
        #img = self.mel_spec_2_log_spec(img)
        return img

    def mel_spec_2_log_spec(self, mel_spectrogram):
        return librosa.core.power_to_db(mel_spectrogram)
        
    
    def preprocess(self,img):
        imarray = self.load_imarray(img)
        imarray = self.pad_to_224(imarray)
        imarray -= np.mean(imarray, keepdims = True)
        imarray /= np.std(imarray, keepdims = True) + tf.keras.backend.epsilon()
        return self.to_rgb(imarray)
    
    def pad_to_224(self, image):
        # image shape: (128, 173) or (128, 173, 1)

        if len(image.shape) == 2:
            image = tf.expand_dims(image, axis=-1)  # (H, W, 1)

        pad_top = 48
        pad_bottom = 48
        pad_left = 25
        pad_right = 26

        padded = tf.pad(
            image,
            paddings=[[pad_top, pad_bottom],
                      [pad_left, pad_right],
                      [0, 0]],
            mode='CONSTANT',
            constant_values=0
        )

        return padded  # (224, 224, 1)
    
    def to_rgb(self, image):
        return tf.repeat(image, repeats=3, axis=-1)  # (224, 224, 3)

    
class Generator(Sequence, Preprocessing):

    
    def __init__(self, Ims, Labels, shp, batch_size, p_arrays):
        
        self.Ims       = Ims
        self.Labels    = Labels
        self.shp = shp
        self.batch_size = batch_size
        self.p_arrays = p_arrays
        
        super().__init__(self.p_arrays)
          
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
    


    
class CricketClassifier:
    def __init__(
        self,
        input_shape=(224, 224, 3),
        num_classes=47,
        use_pretrained=True,
        fine_tune=False,
        fine_tune_at=140,
        dropout_rate=0.2
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.use_pretrained = use_pretrained
        self.fine_tune = fine_tune
        self.fine_tune_at = fine_tune_at
        self.dropout_rate = dropout_rate

        self.base_model = None
        self.model = None

    def _build_base_model(self):
        base_model = ResNet50(
            weights='imagenet' if self.use_pretrained else None,
            include_top=False,
            input_shape=self.input_shape
        )

        base_model.trainable = False
        return base_model

    def unfreeze_top_layers(self, fine_tune_at=None):
        if self.base_model is None:
            raise ValueError("Model not built yet.")

        if fine_tune_at is not None:
            self.fine_tune_at = fine_tune_at

        # Freeze everything first (safe design)
        for layer in self.base_model.layers:
            layer.trainable = False

        # Unfreeze top layers
        for layer in self.base_model.layers[self.fine_tune_at:]:
            layer.trainable = True

    def ClassifyIDModel(self):
        self.base_model = self._build_base_model()

        # Apply fine-tuning if requested
        if self.fine_tune:
            self.unfreeze_top_layers(self.fine_tune_at)

        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)

        # =========================
        # CLASSIFICATION HEAD (your original design)
        # =========================

        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)

        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate / 2)(x)

        outputs = Dense(self.num_classes, activation='softmax')(x)

        self.model = Model(
            inputs=self.base_model.input,
            outputs=outputs,
            name="CricketClassifier"
        )

        return self.model   
          
        
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

#=================================

class LabelEncoder:

    def __init__(self, p_segments, p_samples):

        self.p_segments = p_segments

        self.files = {
            "train": f"{p_samples}n1_train.csv",
            "validation": f"{p_samples}n1_validation.csv",
            "n1_test": f"{p_samples}n1_test.csv",
            "n2_test": f"{p_samples}n2_test.csv",
            "n3_test": f"{p_samples}n3_test.csv",
        }

    def hot_encode_labels(self, samples, id_to_samples):

        sample_to_numeric_id = {sample: i for i, sample_list in enumerate(id_to_samples.values()) for sample in sample_list}
        sample_keys, numeric_ids = zip(*sample_to_numeric_id.items())
        sample_to_label = {sample_keys[i]: label for i, label in enumerate(to_categorical(numeric_ids))}

        return np.array(samples), np.array([sample_to_label[sample] for sample in samples])

    def execution(self):

        df = pd.read_csv(self.p_segments)[["Segment", "ID"]]
        sample_to_id = {sample: ID for sample, ID in df.to_records(index=False)}

        id_to_samples = {}
        for sample, ID in sample_to_id.items(): id_to_samples.setdefault(ID, []).append(sample)

        outputs = []

        for name in ["train", "validation", "n1_test", "n2_test", "n3_test"]:

            samples = pd.read_csv(self.files[name]).Segment.values
            x, y = self.hot_encode_labels(samples, id_to_samples)
            idx = np.random.permutation(len(samples))

            outputs.extend([x[idx], y[idx]])

        return tuple(outputs)

#=========================================================

class Predictions:

    def __init__(self, model, p_preds, data):

        self.model = model
        self.p_preds = p_preds

        self.data = data

    def predict(self, samples, hotlabels, generator, name, verbose=0):

        predictions = self.model.predict(generator, verbose=verbose)

        observed = np.argmax(hotlabels, axis=1)
        predicted = np.argmax(predictions, axis=1)
        maxima = np.max(predictions, axis=1)

        accuracy = round(np.mean(observed == predicted), 3)

        print(f"accuracy for {name} = {accuracy}")

        pd.DataFrame({
            "samples": samples,
            "labels": observed,
            "predictions": predicted,
            "maximum": maxima,
        }).to_csv(f"{self.p_preds}{name}.csv", index=False)

        return accuracy

    def execution(self):

        metrics = {
            name: self.predict(samples, labels, generator, name)
            for name, (samples, labels, generator) in self.data.items()
        }

        df = pd.DataFrame(
            [metrics],
            columns=["validation", "one_n1_test", "one_n2_test", "one_n3_test"],
        )

        df.to_csv(f"{self.p_preds}one_clas_metrics.csv", index=False)

        return df

