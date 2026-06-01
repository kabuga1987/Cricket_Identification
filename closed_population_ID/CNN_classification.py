import os
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from classification_helper import*
from sklearn.metrics import accuracy_score
#==============================================

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

#====================================
if __name__ == "__main__":

    # Paths
    # -----------------------------
    p1 = "../segment_index_extraction/segment_data.csv"
    p2 = "train_val_test_segment_data/"
    p3 = "classification_predictions/CNNs/"
    PATH_ARRAY = "../segment_spectrogram_mfcc_feature_extraction/spectrogram_arrays/"
    PATH_W = "weights/n1_classification.h5"
    os.makedirs(p3, exist_ok=True)
    os.makedirs("weights/", exist_ok=True)
    # -----------------------------
    # Model / Training config
    # -----------------------------
    SHP = (128, 173, 1)
    LR = 1e-3
    FINE_LR = 1e-4
    EPOCHS = 5
    BATCH_SIZE = 128
    USE_MEANVAR = True
    # -----------------------------
    # Label encoding
    # -----------------------------
    encoder = LabelEncoder(p1, p2)

    (
        train_x, train_y,
        val_x, val_y,
        test_x,
        test_y,
        n2_x,
        n2_y,
        n3_x,
        n3_y
    ) = encoder.HotEncodeLabels()

    # -----------------------------
    # Data generators
    # -----------------------------
    Tgen = Generator(train_x, train_y, SHP, BATCH_SIZE, PATH_ARRAY, USE_MEANVAR)
    Vgen = Generator(val_x, val_y, SHP, BATCH_SIZE, PATH_ARRAY, USE_MEANVAR)
    TstGen = Generator(test_x, test_y, SHP, BATCH_SIZE, PATH_ARRAY, USE_MEANVAR)
    N2gen = Generator(n2_x, n2_y, SHP, BATCH_SIZE, PATH_ARRAY, USE_MEANVAR)
    N3gen = Generator(n3_x, n3_y, SHP, BATCH_SIZE, PATH_ARRAY, USE_MEANVAR)
    # -----------------------------
    # Model setup
    # -----------------------------
    architecture = ModelArchitecture(SHP)
    trainer = Training()
    trainer.architecture = architecture
    trainer.Tgenerator = Tgen
    trainer.Vgenerator = Vgen
    model = trainer.train(LR, FINE_LR, PATH_W, EPOCHS)
    # -----------------------------
    # Predictions
    # -----------------------------
    predictor = Predictions(
        model=model,
        v1x=val_x,
        v1y=val_y,
        n1x=test_x,
        n1y=test_y,
        n2x=n2_x,
        n2y=n2_y,
        n3x=n3_x,
        n3y=n3_y,
        vgen=Vgen,
        gen1=TstGen,
        gen2=N2gen,
        gen3=N3gen,
        p_preds=p3
    )
    metrics = predictor.execution()
    print()
    print()
    print(metrics)
