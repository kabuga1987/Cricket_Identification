import os
import numpy as np
import pandas as pd


class TrainValidationTestSplit:

    def __init__(self, path_seg_data, path_trn_val_tst_data, night):
        """
        Parameters
        ----------
        path_seg_data : str
            Path to the original segment data.
        path_trn_val_tst_data : str
            Directory where training, validation, and test CSV files will be saved.
        night : int
            Recording night to use for splitting.
        """

        self.path_seg_data = path_seg_data
        self.path_trn_val_tst_data = path_trn_val_tst_data
        self.night = night
        self.df = self.csv_data()

    def csv_data(self):
        """Read the segment data CSV file."""
        return pd.read_csv(self.path_seg_data)

    def image_to_id(self, df):
        """Map each segment to its corresponding individual ID."""
        return dict(df[["Segment", "ID"]].to_records(index=False))

    def image_to_call(self, df):
        """Map each segment to its corresponding call."""
        return dict(df[["Segment", "Call"]].to_records(index=False))

    def group_elements(self, element_to_group):
        """Group elements by their associated label."""
        group_to_elements = {}
        for element, group in element_to_group.items():
            group_to_elements.setdefault(group, []).append(element)
        return group_to_elements

    def train_val_test_split(self):
        """
        Split segments into training, validation, and test sets.

        Segments are first grouped by individual ID and then by call.
        Within each call, segments are randomly assigned as follows:

        - 60% training
        - 20% validation
        - 20% testing
        """

        train, validation, test = [], [], []
        df_night = self.df[self.df.Night == self.night]
        image_to_call = self.image_to_call(df_night)
        image_to_id = self.image_to_id(df_night)
        id_to_images = self.group_elements(image_to_id)

        for images in id_to_images.values():
            image_to_call_subset = {image: image_to_call[image] for image in images}
            call_to_images = self.group_elements(image_to_call_subset)
            for call_images in call_to_images.values():
                call_images = sorted(call_images)
                indices = np.arange(len(call_images))
                np.random.shuffle(indices)
                call_images = list(np.array(call_images)[indices])
                split_60 = int(len(call_images) * 0.60)
                split_80 = int(len(call_images) * 0.80)
                train.extend(call_images[:split_60])
                validation.extend(call_images[split_60:split_80])
                test.extend(call_images[split_80:])

        return train, validation, test

    def save_train_val_test_segment_data(self):
        """Save training, validation, and test segment metadata."""

        train, validation, test = self.train_val_test_split()

        df_train = self.df[self.df.Segment.isin(train)]
        df_validation = self.df[self.df.Segment.isin(validation)]
        df_test = self.df[self.df.Segment.isin(test)]

        df_train.to_csv(
            os.path.join(self.path_trn_val_tst_data, f"n{self.night}_train.csv"),
            index=False
        )

        df_validation.to_csv(
            os.path.join(self.path_trn_val_tst_data, f"n{self.night}_validation.csv"),
            index=False
        )

        df_test.to_csv(
            os.path.join(self.path_trn_val_tst_data, f"n{self.night}_test.csv"),
            index=False
        )

#=======================================================

if __name__ == "__main__":
    p_seg = "../segment_index_extraction/segment_data.csv"
    p_t_v_t = "train_val_test_segment_data/"
    tvt = TrainValidationTestSplit(p_seg, p_t_v_t,1)
    tvt.save_train_val_test_segment_data()
    # Creation of night 2 and night 3 test sets 
    df = tvt.df
    d2 = df[df.Night == 2]
    d3 = df[df.Night == 3]
    d2.to_csv(p_t_v_t + "n2_test.csv", index = False)
    d3.to_csv(p_t_v_t + "n3_test.csv", index = False)
    print(f"Saving train, validation, and test segment data has been completed.")
        