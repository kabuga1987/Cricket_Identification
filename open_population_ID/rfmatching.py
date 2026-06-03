import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

#===============================================

import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class RandomForest:

    def __init__(self, path_data_pairs, path_mfccs, path_predictions):

        self.path_data_pairs = path_data_pairs
        self.path_mfccs = path_mfccs
        self.path_predictions = path_predictions

        self.p_trn = os.path.join(path_data_pairs, "n1_train_pairs.csv")
        self.p_val = os.path.join(path_data_pairs, "n1_validation_pairs.csv")

        self.test_sets = {
            "n1_test": os.path.join(path_data_pairs, "n1_test_pairs.csv"),
            "n2_test": os.path.join(path_data_pairs, "n2_test_pairs.csv"),
            "n3_test": os.path.join(path_data_pairs, "n3_test_pairs.csv"),
            "n1_n2_test": os.path.join(path_data_pairs, "n1_n2_test_pairs.csv"),
            "n1_n3_test": os.path.join(path_data_pairs, "n1_n3_test_pairs.csv"),
            "n2_n3_test": os.path.join(path_data_pairs, "n2_n3_test_pairs.csv")
        }

    @staticmethod
    def samples_labels(path):

        df = pd.read_csv(path)

        return (
            df,
            df.sample_1.values,
            df.sample_2.values,
            df.label.values
        )

    @staticmethod
    def sample_to_mfccs(path):

        mfccs_df = pd.read_csv(path)

        return {
            row[0]: row[1:].astype(np.float32)
            for row in mfccs_df.values
        }

    @staticmethod
    def samples_to_mfccs(samples, sample_to_mfccs):

        return np.array(
            [sample_to_mfccs[sample] for sample in samples],
            dtype=np.float32
        )

    def combine(self, sample_to_mfccs, path_pairs):

        df, sample_1, sample_2, labels = self.samples_labels(path_pairs)

        mfccs_1 = self.samples_to_mfccs(sample_1, sample_to_mfccs)
        mfccs_2 = self.samples_to_mfccs(sample_2, sample_to_mfccs)

        features = np.hstack([mfccs_1, mfccs_2])

        assert features.shape[0] == labels.shape[0]
        assert features.shape[1] == 2 * mfccs_1.shape[1]

        return df, features, labels

    @staticmethod
    def evaluation_metrics(y_true, y_pred):

        return (
            round(precision_score(y_true, y_pred), 4),
            round(recall_score(y_true, y_pred), 4),
            round(f1_score(y_true, y_pred), 4),
            round(accuracy_score(y_true, y_pred), 4)
        )

    def train_model(self, sample_to_mfccs, path_pairs):

        _, features, labels = self.combine(sample_to_mfccs, path_pairs)

        model = RandomForestClassifier(
            n_estimators=1000,
            max_depth=None,
            max_features=5,
            n_jobs=-1,
            random_state=42
        )

        model.fit(features, labels)

        return model

    def predict(self, model, sample_to_mfccs, path_pairs, dataset_name):

        df, features, labels = self.combine(sample_to_mfccs, path_pairs)

        predictions = model.predict(features)

        df["predictions"] = predictions

        df.to_csv(
            os.path.join(
                self.path_predictions,
                f"{dataset_name}.csv"
            ),
            index=False
        )

        return self.evaluation_metrics(labels, predictions)

    def execution(self):

        sample_to_mfccs = self.sample_to_mfccs(self.path_mfccs)

        model = self.train_model(
            sample_to_mfccs,
            self.p_trn
        )

        metrics = {}

        metrics["n1_val"] = self.predict(
            model,
            sample_to_mfccs,
            self.p_val,
            "n1_val"
        )

        for name, path in self.test_sets.items():

            metrics[name] = self.predict(
                model,
                sample_to_mfccs,
                path,
                name
            )

        df_metrics = pd.DataFrame.from_dict(
            metrics,
            orient="index",
            columns=[
                "precision",
                "recall",
                "f1_score",
                "accuracy"
            ]
        )

        df_metrics.index.name = "dataset"

        df_metrics.to_csv(
            os.path.join(
                self.path_predictions,
                "one_metrics.csv"
            )
        )

        return df_metrics
#======================================

if __name__ == "__main__":

    rf = RandomForest(
        path_data_pairs="pairs/",
        path_mfccs="../segment_spectrogram_mfcc_feature_extraction/mfcc_data.csv",
        path_predictions="matching_predictions/RFs/"
    )

    df_metrics = rf.execution()

    print("\n" + "=" * 50)
    print("Evaluation Metrics")
    print("=" * 50 + "\n")
    print(df_metrics)
        
    
               