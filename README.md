## Cricket Individual animal identification
This project integrates preprocessing, feature extraction, and identification steps to ascertain the feasibility of identifying cricket individuals exclusively from their calls.

1. The preprocessing step ([chirp_index_extraction](https://github.com/kabuga1987/Cricket_Identification/tree/main/chirp_index_extraction) or [segment_index_extraction](https://github.com/kabuga1987/Cricket_Identification/tree/main/segment_index_extraction)) applies a noise reduction algorithm and a lower filter to remove noise before segmenting the cricket signal into acoustically relevant units.
2. The [feature extraction](https://github.com/kabuga1987/Cricket_Identification/tree/main/segment_spectrogram_mfcc_feature_extraction) step generates spectrograms, MFCCs, and temporal features from the segmented acoustic units.
3. The identification step utilizes standard classification models, such as Random Forests (RFs) and Convolutional Neural Networks (CNNs), to address cricket individual identification in a [closed population setting](https://github.com/kabuga1987/Cricket_Identification/tree/main/closed_population_ID), where the number of calling individuals is known. In this case, the task is to assign a call to one of the existing individuals. The models process input data (e.g., MFCC vectors for RFs or spectrograms for CNNs) to predict the associated individual categories. In contrast, for [open population identification](https://github.com/kabuga1987/Cricket_Identification/tree/main/open_population_ID), where the number of individuals is unknown and may vary over time, similarity learning models are employed. These models work by processing input pairs (e.g., pairs of MFCCs for RFs or spectrograms for CNNs) to determine whether the calls were made by the same individual or different individuals.

This code accompanies the paper: **Individual identification of field crickets from their calls using similarity comparison networks**

## Authors
Emmanuel Kabuga, Diptarup Nandi, Stuart Burrell, Gciniwe Dlamini, Rohini Balakrishnan, Bubacarr Bah, Ian Durbach

## Requirements


All required packages can be installed  using `pip install -r requirements.txt`

Numpy

Pandas

Sklearn

Tensorflow

Keras

Librosa

Scipy



