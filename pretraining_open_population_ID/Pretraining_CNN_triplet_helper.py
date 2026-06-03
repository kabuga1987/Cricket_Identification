import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    BatchNormalization,
    Dropout
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
from sklearn.metrics import precision_recall_curve
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import os
import matplotlib.pyplot as plt

#========================================
class DataLoader:

    def __init__(self, path_train, path_validation, samples_per_ID):
        self.path_train = path_train
        self.path_validation = path_validation
        self.samples_per_ID = samples_per_ID

    @staticmethod
    def slice_data_frame(path):
        return pd.read_csv(path)[["Segment", "ID"]]

    @staticmethod
    def sample_to_ID(df):
        return dict(df.to_records(index=False))

    @staticmethod
    def ID_to_samples(sample_2_ID):
        ID_2_samples = {}
        for sample, ID in sample_2_ID.items():
            ID_2_samples.setdefault(ID, []).append(sample)
        return ID_2_samples

    def row_per_ID_matrix(self, ID_2_samples):
        grouped_samples = list(ID_2_samples.values())
        max_samples = max(map(len, grouped_samples))
        ID_matrix = np.zeros((len(grouped_samples), max_samples), dtype="<U22")
        for i, samples in enumerate(grouped_samples):
            ID_matrix[i] = np.hstack([samples, np.zeros(max_samples - len(samples))])
        return ID_matrix

    def list_of_all_samples(self, ID_matrix):
        ordered_samples = []
        for i in np.arange(0, ID_matrix.shape[1], self.samples_per_ID):
            batch = ID_matrix[:, i:i + self.samples_per_ID].flatten()
            ordered_samples.extend(batch.tolist())
        ordered_samples = np.array(ordered_samples)
        return ordered_samples[ordered_samples != "0.0"]

    @staticmethod
    def sample_2_label_ID(samples, ID_2_samples):
        sample_2_label = {sample: i for i, values in enumerate(ID_2_samples.values()) for sample in values}
        return zip(*[(sample, sample_2_label[sample]) for sample in samples])

    def execution(self, path):
        df = self.slice_data_frame(path)
        sample_2_ID = self.sample_to_ID(df)
        ID_2_samples = self.ID_to_samples(sample_2_ID)
        ID_matrix = self.row_per_ID_matrix(ID_2_samples)
        ordered_samples = self.list_of_all_samples(ID_matrix)
        samples, labels = self.sample_2_label_ID(ordered_samples, ID_2_samples)
        return np.array(samples), np.array(labels)

    def train_validation_samples_labels(self):
        train_samples, train_labels = self.execution(self.path_train)
        val_samples, val_labels = self.execution(self.path_validation)
        return train_samples, train_labels, val_samples, val_labels
        
#===============================================================

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
        top, bottom, left, right = 48,48, 25, 26
        padded = tf.pad(image, paddings=[[top, bottom], [left,right],[0, 0]], mode='CONSTANT', constant_values=0)
        return padded  # (224, 224, 1)
    
    def to_rgb(self, image):
        return tf.repeat(image, repeats=3, axis=-1)  # (224, 224, 3)
  
#=======================================================

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
        ImsL  = np.zeros((size,), dtype = K.floatx())
        for i in range(size):
            ImsA[i,:,:,:] = super().preprocess(self.Ims[start + i])
            ImsL[i]     = self.Labels[start+i]
            
        return ImsA, ImsL
    
    def __len__(self):
        return (len(self.Ims) + self.batch_size - 1)//self.batch_size


class TestGe(Sequence, Preprocessing):

    
    def __init__(self, TeIs, shp, batch_size, p_arrays):
        self.TeIs = TeIs
        self.shp = shp
        self.batch_size = batch_size
        self.p_arrays = p_arrays
        super().__init__(self.p_arrays)
          
    def __getitem__(self, index):
        start = self.batch_size*index
        size  = min(len(self.TeIs) - start, self.batch_size)
        ImsA  = np.zeros((size,) + self.shp, dtype=K.floatx())
        for i in range(size):
            ImsA[i,:,:,:] = super().preprocess(self.TeIs[start + i])
        return ImsA
    
    def __len__(self):
        return (len(self.TeIs) + self.batch_size - 1)//self.batch_size

#================================================

class Evaluation:

    @staticmethod
    def compute_dist(Lembs, Rembs):
        return np.linalg.norm(Lembs - Rembs, axis=1)

    @staticmethod
    def calc_acc(distances, Y, threshold):
        preds = np.where(distances <= np.abs(threshold), 1, 0)
        return np.mean(Y == preds)

    @staticmethod
    def prec_recal_thr(distances, Y):
        return precision_recall_curve(Y, -distances)

    @staticmethod
    def optimized_metrics(precision, recall, thresholds):
        f1_scores = 2 * precision * recall / (precision + recall + K.epsilon())
        idx = np.argmax(f1_scores)
        return (round(precision[idx], 3), round(recall[idx], 3), round(f1_scores[idx], 3), round(np.abs(thresholds[idx]), 3))

    @staticmethod
    def plot_distances(distances, Y, threshold, recall, precision, f1, accuracy, night):
        matches = distances[Y == 1]
        non_matches = distances[Y == 0]
        x1 = np.arange(len(non_matches))
        x2 = np.arange(len(non_matches), len(non_matches) + len(matches))
        plt.figure(figsize=(12, 6))
        plt.scatter(x1, non_matches, color = 'r', label="Non-matches")
        plt.scatter(x2, matches, color = 'g', label="Matches")
        plt.axhline(y=np.abs(threshold), linestyle="-", label="Threshold")
        plt.title(f"Night = {night}, Recall = {recall}, Precision = {precision}, F1 = {f1}, Accuracy = {accuracy}")
        plt.xlabel("Number of pairs")
        plt.ylabel("Distance between image pairs")
        plt.legend()
        plt.savefig(f"figures/{night}.pdf")

    def evaluation_metrics(self, Lembs, Rembs, Y, night):
        distances = self.compute_dist(Lembs, Rembs)
        precision, recall, thresholds = self.prec_recal_thr(distances, Y)
        precision_opt, recall_opt, f1_opt, threshold_opt = self.optimized_metrics(precision, recall, thresholds)
        accuracy_opt = round(self.calc_acc(distances, Y, threshold_opt), 4)
        self.plot_distances(distances, Y, threshold_opt, recall_opt, precision_opt, f1_opt, accuracy_opt, night)
        preds = np.where(distances <= threshold_opt, 1, 0)
        return (precision_opt, recall_opt, f1_opt, accuracy_opt, threshold_opt, distances, preds)

    def mymetrics(self, embsL, embsR, Y, threshold, night):
        distances = self.compute_dist(embsL, embsR)
        preds = np.where(distances <= np.abs(threshold), 1, 0)
        TP = np.sum((Y == 1) & (preds == 1))
        TN = np.sum((Y == 0) & (preds == 0))
        FP = np.sum((Y == 0) & (preds == 1))
        FN = np.sum((Y == 1) & (preds == 0))
        recall = round(TP / (TP + FN + K.epsilon()), 4)
        precision = round(TP / (TP + FP + K.epsilon()), 4)
        f1 = round(2 * precision * recall / (precision + recall + K.epsilon()), 4)
        accuracy = round((TP + TN) / (TP + TN + FP + FN + K.epsilon()), 4)
        self.plot_distances(distances, Y, threshold, recall, precision, f1, accuracy, night)
        return precision, recall, f1, accuracy, distances, preds

#====================================================================

class CricketEmbedding:
    def __init__(
        self,
        input_shape=(224, 224, 3),
        use_pretrained=True,
        fine_tune=True,
        fine_tune_at=140
    ):
        self.input_shape = input_shape
        self.use_pretrained = use_pretrained
        self.fine_tune = fine_tune
        self.fine_tune_at = fine_tune_at


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

    def embedding_model(self):
        self.base_model = self._build_base_model()

        # Apply fine-tuning if requested
        if self.fine_tune:
            self.unfreeze_top_layers(self.fine_tune_at)

        x = self.base_model.output
        outputs = GlobalAveragePooling2D()(x)

        self.model = Model(
            inputs=self.base_model.input,
            outputs=outputs,
            name="CricketEmbedding"
        )
        
        return self.model

#=========================================================

class Training:
    
    def __init__(self):
        
        self.architecture = None
        self.Tgenerator = None
        self.Vgenerator = None

    def train(self, Lr, lr, pw, Epochs):

        model = self.architecture.embedding_model()
        model.compile(loss=tfa.losses.TripletSemiHardLoss(), optimizer=Adam(Lr))

        callbacks = [
            ModelCheckpoint(
                filepath=f"{pw}CNN_triplet_single_night.h5",
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                patience=5,
                factor=0.5,
                min_lr=lr,
                verbose=0
            )
        ]

        print(f"="*50)
        print(f"Training has started")
        print(f"="*50)

        history = model.fit(
            self.Tgenerator,
            validation_data=self.Vgenerator,
            epochs=Epochs,
            callbacks=callbacks,
            max_queue_size=12,
            workers=6,
            verbose=1
        )

        print(f"="*50)
        print(f"Loading best model weights")
        print(f"="*50)

        model.load_weights(f"{pw}CNN_triplet_single_night.h5")

        val_loss = model.evaluate(self.Vgenerator, verbose=0)

        print(f"Validation loss : {val_loss}")
        print(f"="*50)

        return model, history

#=====================================================

class Predictions:

    def __init__(self, path_val_test_samples, path_tests, p_arrays, path_preds, shp, bs):

        self.path_val_test_samples = path_val_test_samples
        self.path_tests = path_tests
        self.p_arrays = p_arrays
        self.path_preds = path_preds
        self.bs = bs
        self.shp = shp

        self.evaluator = None
        self.model = None

        self.test_files = {
            "n1_validation": os.path.join(path_tests, "n1_validation_pairs.csv"),
            "n1_test": os.path.join(path_tests, "n1_test_pairs.csv"),
            "n2_test": os.path.join(path_tests, "n2_test_pairs.csv"),
            "n3_test": os.path.join(path_tests, "n3_test_pairs.csv"),
            "n1_n2_test": os.path.join(path_tests, "n1_n2_test_pairs.csv"),
            "n1_n3_test": os.path.join(path_tests, "n1_n3_test_pairs.csv"),
            "n2_n3_test": os.path.join(path_tests, "n2_n3_test_pairs.csv"),
        }

        self.val_test_files = [
            os.path.join(path_val_test_samples, "n1_validation.csv"),
            os.path.join(path_val_test_samples, "n1_test.csv"),
            os.path.join(path_val_test_samples, "n2_test.csv"),
            os.path.join(path_val_test_samples, "n3_test.csv"),
        ]

    # =========================================================
    # LOAD VALIDATION + TEST SAMPLES
    # =========================================================
    def evaluation_samples(self):
        return pd.concat([pd.read_csv(f) for f in self.val_test_files]).Segment.values

    # =========================================================
    # EMBEDDINGS
    # =========================================================
    def sample_to_embedding(self, samples):
        testgen = TestGe(samples, self.shp, self.bs, self.p_arrays)
        emb = self.model.predict(testgen, verbose=1)
        return {samples[i]: emb[i] for i in range(len(samples))}

    # =========================================================
    # VALIDATION THRESHOLD
    # =========================================================
    def optimal_threshold(self, emb_dict, name):

        df = pd.read_csv(self.test_files["n1_validation"])
        s1, s2, y = df.sample_1.values, df.sample_2.values, df.label.values

        e1 = np.array([emb_dict[s] for s in s1])
        e2 = np.array([emb_dict[s] for s in s2])

        pre, rec, f1, acc, thr, dist, preds = self.evaluator.evaluation_metrics(e1, e2, y, name)

        df["Euclid_dist"], df["Threshold"], df["Predictions"] = dist, thr, preds
        df.to_csv(os.path.join(self.path_preds, name + ".csv"), index=False)

        return pre, rec, f1, acc, thr

    # =========================================================
    # TEST EVALUATION
    # =========================================================
    def test_predictions(self, path, emb_dict, thr, name):

        df = pd.read_csv(path)
        s1, s2, y = df.sample_1.values, df.sample_2.values, df.label.values

        e1 = np.array([emb_dict[s] for s in s1])
        e2 = np.array([emb_dict[s] for s in s2])

        P, R, F, A, dist, preds = self.evaluator.mymetrics(e1, e2, y, thr, name)

        df["Euclid_dist"], df["Threshold"], df["Predictions"] = dist, thr, preds
        df.to_csv(os.path.join(self.path_preds, name + ".csv"), index=False)

        return P, R, F, A

    # =========================================================
    # FULL EXECUTION PIPELINE
    # =========================================================
    def Execution(self):

        samples = self.evaluation_samples()
        emb_dict = self.sample_to_embedding(samples)

        pre, rec, f1, acc, thr = self.optimal_threshold(emb_dict, "n1_val")

        results = {"n1v": [pre, rec, f1, acc]}

        for name, path in self.test_files.items():
            if name.endswith("_test"):
                results[name.replace("_test", "")] = self.test_predictions(
                    path, emb_dict, thr, name
                )

        df = pd.DataFrame.from_dict(
            results,
            orient="index",
            columns=["precision", "recall", "f1_score", "accuracy"]
        ).rename_axis("nights")

        df.to_csv(os.path.join(self.path_preds, "metrics.csv"))
        print("Evaluation has finished")

        return df
