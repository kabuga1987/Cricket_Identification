import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import Add, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers, backend as K
import tensorflow_addons as tfa

# =========================================================
# Data loading
# =========================================================

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


# =========================================================
# Preprocessing
# =========================================================

class Preprocessing:

    def __init__(self, path_2_imarray, meanvar=True):
        self.path_2_imarray = path_2_imarray
        self.meanvar = meanvar

    def load_imarray(self, img):
        img = np.load(f"{self.path_2_imarray}{img}.npy")
        return np.expand_dims(img, axis=2)

    def zeromean_unitvariance_normalizer(self, img):
        x = self.load_imarray(img)
        x -= np.mean(x, keepdims=True)
        x /= np.std(x, keepdims=True) + K.epsilon()
        return x

    def min_max_normalizer(self, img, _min=0, _max=1):
        x = self.load_imarray(img)
        x = (x - x.min()) / (x.max() - x.min())
        x = x * (_max - _min) + _min
        return x

    def preprocess(self, img):
        if self.meanvar:
            return self.zeromean_unitvariance_normalizer(img)
        return self.min_max_normalizer(img)

# =========================================================
# Generators
# =========================================================

class Generator(Sequence, Preprocessing):

    def __init__(self, Ims, Labels, shp, batch_size, p_arrays, meanvar=True):
        self.Ims = Ims
        self.Labels = Labels
        self.shp = shp
        self.batch_size = batch_size
        super().__init__(p_arrays, meanvar)

    def __getitem__(self, index):
        start = self.batch_size * index
        stop = min(start + self.batch_size, len(self.Ims))
        size = stop - start
        X = np.zeros((size,) + self.shp, dtype=K.floatx())
        y = np.zeros(size, dtype=K.floatx())
        for i, idx in enumerate(range(start, stop)):
            X[i] = self.preprocess(self.Ims[idx])
            y[i] = self.Labels[idx]
        return X, y

    def __len__(self):
        return (len(self.Ims) + self.batch_size - 1) // self.batch_size


class TestGe(Sequence, Preprocessing):

    def __init__(self, TeIs, shp, batch_size, p_arrays, meanvar=True):
        self.TeIs = TeIs
        self.shp = shp
        self.batch_size = batch_size
        super().__init__(p_arrays, meanvar)

    def __getitem__(self, index):
        start = self.batch_size * index
        stop = min(start + self.batch_size, len(self.TeIs))
        size = stop - start
        X = np.zeros((size,) + self.shp, dtype=K.floatx())
        for i, idx in enumerate(range(start, stop)):
            X[i] = self.preprocess(self.TeIs[idx])
        return X

    def __len__(self):
        return (len(self.TeIs) + self.batch_size - 1) // self.batch_size


# =========================================================
# Evaluation
# =========================================================

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


# =========================================================
# Architecture
# =========================================================

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
 
# =========================================================
# Training
# =========================================================

class Training:

    def __init__(self):
        self.architecture = None
        self.Tgenerator = None
        self.Vgenerator = None

    def train(self, max_lr, min_lr, weights_path, epochs):

        model = self.architecture.EmbeddingModel()

        model.compile(
            loss=tfa.losses.TripletSemiHardLoss(),
            optimizer=Adam(max_lr)
        )

        callbacks = [
            ModelCheckpoint(
                weights_path,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True
            ),

            ReduceLROnPlateau(
                monitor="val_loss",
                patience=5,
                factor=0.5,
                min_lr=min_lr,
                verbose=0
            )
        ]

        print("="*50)
        print("Training model has started")
        print("="*50)

        model.fit(
            self.Tgenerator,
            validation_data=self.Vgenerator,
            epochs=epochs,
            callbacks=callbacks,
            max_queue_size=12,
            workers=6,
            verbose=2
        )

        model.load_weights(weights_path)

        return model

#================
# predicting
#================

class Predictions:

    def __init__(self, path_val_test_samples, path_val_samples, path_val_pairs, path_tests, p_arrays, p_preds, shp, bs):

        self.path_val_test_samples = path_val_test_samples
        self.path_val_samples = path_val_samples
        self.path_val_pairs = path_val_pairs
        self.path_preds = p_preds

        self.test_files = {
            "n1_test": f"{path_tests}n1_test_pairs.csv",
            "n2_test": f"{path_tests}n2_test_pairs.csv",
            "n3_test": f"{path_tests}n3_test_pairs.csv",
            "n1_n2_test": f"{path_tests}n1_n2_test_pairs.csv",
            "n1_n3_test": f"{path_tests}n1_n3_test_pairs.csv",
            "n2_n3_test": f"{path_tests}n2_n3_test_pairs.csv",
        }

        self.p_arrays = p_arrays
        self.shp = shp
        self.bs = bs

        self.evaluator = None
        self.model = None

    def evaluation_samples(self):

        df = pd.concat([
            pd.read_csv(self.path_val_samples),
            pd.read_csv(f"{self.path_val_test_samples}n1_test.csv"),
            pd.read_csv(f"{self.path_val_test_samples}n2_test.csv"),
            pd.read_csv(f"{self.path_val_test_samples}n3_test.csv"),
        ]).drop_duplicates("Segment")

        return df.Segment.values

    def sample_to_embedding(self, samples, model):

        generator = TestGe(samples, self.shp, self.bs, self.p_arrays, meanvar=True)

        embeddings = model.predict(generator, verbose=1, workers=6, use_multiprocessing=True)

        return dict(zip(samples, embeddings))

    @staticmethod
    def extract_embeddings(samples_1, samples_2, sample_to_embedding):

        return (
            np.stack([sample_to_embedding[sample] for sample in samples_1]),
            np.stack([sample_to_embedding[sample] for sample in samples_2]),
        )

    def save_predictions(self, df, distances, threshold, preds, name):

        df["Euclid_dist"] = distances
        df["Threshold"] = threshold
        df["Predictions"] = preds

        df.to_csv(f"{self.path_preds}{name}.csv", index=False)

    def optimal_threshold(self, path_pairs, sample_to_embedding, name):

        df = pd.read_csv(path_pairs)

        e1, e2 = self.extract_embeddings(
            df.sample_1.values,
            df.sample_2.values,
            sample_to_embedding,
        )

        precision, recall, f1, accuracy, threshold, distances, preds = self.evaluator.evaluation_metrics(
            e1, e2, df.label.values, name
        )

        self.save_predictions(df, distances, threshold, preds, name)

        return precision, recall, f1, accuracy, threshold

    def test_predictions(self, path_pairs, sample_to_embedding, threshold, name):

        df = pd.read_csv(path_pairs)

        e1, e2 = self.extract_embeddings(
            df.sample_1.values,
            df.sample_2.values,
            sample_to_embedding,
        )

        precision, recall, f1, accuracy, distances, preds = self.evaluator.mymetrics(
            e1, e2, df.label.values, threshold, name
        )

        self.save_predictions(df, distances, threshold, preds, name)

        return precision, recall, f1, accuracy

    def execution(self, model):

        samples = self.evaluation_samples()

        sample_to_embedding = self.sample_to_embedding(samples, model)

        pre, rec, f1, acc, thr = self.optimal_threshold(
            self.path_val_pairs,
            sample_to_embedding,
            "n1_val",
        )

        metrics = {"n1v": [pre, rec, f1, acc]}

        for name, path in self.test_files.items():

            metrics[name.replace("_test", "")] = self.test_predictions(
                path,
                sample_to_embedding,
                thr,
                name,
            )

        df = pd.DataFrame.from_dict(
            metrics,
            orient="index",
            columns=["precision", "recall", "f1_score", "accuracy"],
        )

        df.rename_axis("nights").to_csv(f"{self.path_preds}one_metrics.csv")

        print("Evaluation has finished")

        return df

# class Predictions:

#     def __init__(self, path_val_test_samples, path_val_samples, path_val_pairs, path_tests, p_arrays, path_preds, shp, bs):

#         # Paths
#         self.path_val_test_samples = path_val_test_samples
#         self.path_val_samples = path_val_samples
#         self.path_val_pairs = path_val_pairs
#         self.path_tests = path_tests
#         self.path_preds = path_preds
#         # Test pair files
#         self.n1_test = f"{path_tests}n1_test_pairs.csv"
#         self.n2_test = f"{path_tests}n2_test_pairs.csv"
#         self.n3_test = f"{path_tests}n3_test_pairs.csv"
#         self.n1_n2_test = f"{path_tests}n1_n2_test_pairs.csv"
#         self.n1_n3_test = f"{path_tests}n1_n3_test_pairs.csv"
#         self.n2_n3_test = f"{path_tests}n2_n3_test_pairs.csv"
#         # Generator parameters
#         self.p_arrays = p_arrays
#         self.shp = shp
#         self.bs = bs
#         # External objects
#         self.evaluator = None
#         self.model = None

#     def evaluation_samples(self, path_test, path_val):
#         # Load validation and test samples
#         df = pd.concat([
#             pd.read_csv(path_val),
#             pd.read_csv(f"{path_test}n1_test.csv"),
#             pd.read_csv(f"{path_test}n2_test.csv"),
#             pd.read_csv(f"{path_test}n3_test.csv")
#         ]).drop_duplicates("Segment")

#         return df.Segment.values

#     def sample_to_embedding(self, samples, model):
#         # Create inference generator
#         testgen = TestGe(samples, self.shp, self.bs, self.p_arrays, meanvar=True)
#         # Compute embeddings
#         embeddings = model.predict(testgen, verbose=1, workers=6, use_multiprocessing=True)
#         # Map samples to embeddings
#         return dict(zip(samples, embeddings))

#     @staticmethod
#     def extract_embeddings(samples_1, samples_2, sample_2_embedding):
#         # Retrieve left and right embeddings
#         embeds_1 = np.stack([sample_2_embedding[sample] for sample in samples_1])
#         embeds_2 = np.stack([sample_2_embedding[sample] for sample in samples_2])
#         return embeds_1, embeds_2

#     def save_predictions(self, df, distances, threshold, preds, night):
#         # Store predictions
#         df["Euclid_dist"] = distances
#         df["Threshold"] = threshold
#         df["Predictions"] = preds
#         # Export predictions
#         df.to_csv(f"{self.path_preds}{night}.csv", index=False)

#     def optimal_threshold(self, path_pairs, sample_2_embedding, night):
#         # Load validation pairs
#         df_pairs = pd.read_csv(path_pairs)
#         # Extract samples and labels
#         samples_1 = df_pairs.sample_1.values
#         samples_2 = df_pairs.sample_2.values
#         labels = df_pairs.label.values
#         # Retrieve embeddings
#         embeds_1, embeds_2 = self.extract_embeddings(samples_1, samples_2, sample_2_embedding)
#         # Compute validation metrics and optimal threshold
#         precision, recall, f1, accuracy, threshold, distances, preds = self.evaluator.evaluation_metrics(
#             embeds_1,
#             embeds_2,
#             labels,
#             night
#         )

#         # Save predictions
#         self.save_predictions(df_pairs, distances, threshold, preds, night)
#         return precision, recall, f1, accuracy, threshold

#     def test_predictions(self, path_pairs, sample_2_embedding, threshold, night):
#         # Load test pairs
#         df_pairs = pd.read_csv(path_pairs)
#         # Extract samples and labels
#         samples_1 = df_pairs.sample_1.values
#         samples_2 = df_pairs.sample_2.values
#         labels = df_pairs.label.values
#         # Retrieve embeddings
#         embeds_1, embeds_2 = self.extract_embeddings(samples_1, samples_2, sample_2_embedding)
#         # Compute metrics
#         precision, recall, f1, accuracy, distances, preds = self.evaluator.mymetrics(
#             embeds_1,
#             embeds_2,
#             labels,
#             threshold,
#             night
#         )

#         # Save predictions
#         self.save_predictions(df_pairs, distances, threshold, preds, night)

#         return precision, recall, f1, accuracy

#     def Execution(self, model):

#         # Retrieve all validation and test samples
#         val_test_samples = self.evaluation_samples(self.path_val_test_samples, self.path_val_samples)
#         # Generate embeddings
#         sample_2_embedding = self.sample_to_embedding(val_test_samples, model)
#         # Compute validation threshold
#         pre, rec, f1s, acc, thr = self.optimal_threshold(self.path_val_pairs, sample_2_embedding, "n1_val")
#         # Evaluate intra-night performance
#         p1, r1, f1, a1 = self.test_predictions(self.n1_test, sample_2_embedding, thr, "n1_test")
#         p2, r2, f2, a2 = self.test_predictions(self.n2_test, sample_2_embedding, thr, "n2_test")
#         p3, r3, f3, a3 = self.test_predictions(self.n3_test, sample_2_embedding, thr, "n3_test")
#         # Evaluate cross-night performance
#         p12, r12, f12, a12 = self.test_predictions(self.n1_n2_test, sample_2_embedding, thr, "n1_n2_test")
#         p13, r13, f13, a13 = self.test_predictions(self.n1_n3_test, sample_2_embedding, thr, "n1_n3_test")
#         p23, r23, f23, a23 = self.test_predictions(self.n2_n3_test, sample_2_embedding, thr, "n2_n3_test")
#         # Store metrics
#         metrics = {
#             "n1v": [pre, rec, f1s, acc],
#             "n11": [p1, r1, f1, a1],
#             "n22": [p2, r2, f2, a2],
#             "n33": [p3, r3, f3, a3],
#             "n12": [p12, r12, f12, a12],
#             "n13": [p13, r13, f13, a13],
#             "n23": [p23, r23, f23, a23]
#         }

#         # Create metrics dataframe
#         df = pd.DataFrame.from_dict(metrics, orient="index", columns=["precision", "recall", "f1_score", "accuracy"])
#         # Name index column
#         df = df.rename_axis("nights")
#         # Export metrics
#         df.to_csv(f"{self.path_preds}one_metrics.csv")
#         print("Evaluation has finished")

#         return df