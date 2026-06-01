import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class SampleData:

    def __init__(self, path_csv, path_samples):

        self.path_csv = path_csv

        self.n1_train = os.path.join(path_samples, "n1_train.csv")
        self.n1_validation = os.path.join(path_samples, "n1_validation.csv")
        self.n1_test = os.path.join(path_samples, "n1_test.csv")
        self.n2_test = os.path.join(path_samples, "n2_test.csv")
        self.n3_test = os.path.join(path_samples, "n3_test.csv")

    def slice_data_frame(self, path):
        """Read a CSV file and retain only Segment and ID columns."""

        return pd.read_csv(path)[["Segment", "ID"]]

    def sample_to_id(self, df):
        """Map each sample to its corresponding individual ID."""

        return dict(df.to_records(index=False))

    def id_to_samples(self, sample_to_id_dict):
        """Group samples by individual ID."""

        id_to_samples = {}
        for sample, individual_id in sample_to_id_dict.items():
            id_to_samples.setdefault(individual_id, []).append(sample)
        return id_to_samples

    def get_samples(self, path):
        """Load sample names from a CSV file."""

        return pd.read_csv(path).Segment.values

    def get_samples_labels(self, samples, id_to_samples_dict):
        """Convert sample names to numeric labels."""

        sample_to_numeric_id = {
            sample: label
            for label, sample_list in enumerate(id_to_samples_dict.values())
            for sample in sample_list
        }

        samples, labels = zip(*[(sample, sample_to_numeric_id[sample]) for sample in samples])

        return np.array(samples), np.array(labels)

    def labelled_samples(self):
        """Load and label all train, validation, and test datasets."""

        df = self.slice_data_frame(self.path_csv)

        sample_to_id = self.sample_to_id(df)
        id_to_samples = self.id_to_samples(sample_to_id)

        datasets = {
            "train": self.get_samples(self.n1_train),
            "validation": self.get_samples(self.n1_validation),
            "night1": self.get_samples(self.n1_test),
            "night2": self.get_samples(self.n2_test),
            "night3": self.get_samples(self.n3_test)
        }

        output = {}

        for dataset_name, samples in datasets.items():
            x, y = self.get_samples_labels(samples, id_to_samples)
            idx = np.random.permutation(len(x))

            output[dataset_name] = {
                "samples": x[idx],
                "labels": y[idx]
            }

        return output


class RFClassifier:

    def __init__(self, p_mfccs, p_predictions, sample_data):

        self.p_mfccs = p_mfccs
        self.p_predictions = p_predictions
        self.sample_data = sample_data

    def sample_to_mfccs(self, path):
        """Map each sample to its MFCC feature vector."""

        mfcc_df = pd.read_csv(path)

        return {row[0]: row[1:] for row in mfcc_df.itertuples(index=False, name=None)}

    def samples_to_mfccs(self, samples, labels, sample_to_mfccs_dict):
        """Retrieve MFCC feature vectors corresponding to a list of samples."""

        mfccs = [sample_to_mfccs_dict[sample] for sample in samples]
        df = pd.DataFrame({"samples": samples,"labels": labels})
        return df, labels, np.asarray(mfccs)

    def random_forest_model(self, sample_to_mfccs_dict, samples, labels):
        """Train a random forest classifier."""

        _, y_train, x_train = self.samples_to_mfccs(samples, labels, sample_to_mfccs_dict)

        model = RandomForestClassifier(
            n_estimators=1000,
            max_features=5
        )

        model.fit(x_train, y_train)

        return model

    def predict(self, model, samples, labels, sample_to_mfccs_dict, dataset_name):
        """Generate predictions and save them to disk."""

        df, y_true, x = self.samples_to_mfccs(samples, labels, sample_to_mfccs_dict)
        predictions = model.predict(x)
        accuracy = np.mean(y_true == predictions)
        df["predictions"] = predictions

        df.to_csv(
            os.path.join(self.p_predictions, f"{dataset_name}.csv"),
            index=False
        )

        return accuracy

    def execution(self):
        """Train the model and evaluate it on all datasets."""

        sample_to_mfccs = self.sample_to_mfccs(self.p_mfccs)
        datasets = self.sample_data.labelled_samples()
        model = self.random_forest_model(
            sample_to_mfccs,
            datasets["train"]["samples"],
            datasets["train"]["labels"]
        )

        accuracies = {}

        for dataset_name in ["validation", "night1", "night2", "night3"]:

            accuracies[dataset_name] = self.predict(
                model=model,
                samples=datasets[dataset_name]["samples"],
                labels=datasets[dataset_name]["labels"],
                sample_to_mfccs_dict=sample_to_mfccs,
                dataset_name=dataset_name
            )

        results = pd.DataFrame([accuracies],index=["accuracy"])

        results.to_csv(
            os.path.join(
                self.p_predictions,
                "single_night_data_accuracy.csv"
            )
        )

        return results
#===================================
if __name__ == "__main__":
    p1 = "../segment_index_extraction/segment_data.csv"
    p2 = "train_val_test_segment_data/"
    sample_data = SampleData(path_csv=p1, path_samples=p2)
    p3 = "../segment_spectrogram_mfcc_feature_extraction/mfcc_data.csv"
    p4 = "classification_predictions/RFs/"

    rf_classifier = RFClassifier(p_mfccs=p3, p_predictions=p4, sample_data=sample_data)

    results = rf_classifier.execution()

    print(results)