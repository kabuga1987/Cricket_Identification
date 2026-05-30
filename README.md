# Cricket individual identification
## Introduction
This project develops deep learning and random forest classifiers for identifying individual crickets in both closed- and open-population settings.

In the [closed-population identification](https://github.com/kabuga1987/Cricket_Identification/tree/main/closed_population_ID), the number of individuals is assumed to be known and fixed throughout the study period. Under this setting, the classifier takes a spectrogram as input for convolutional neural networks (CNNs) or an MFCC feature vector for random forests (RFs) and predicts the probability that the sample belongs to each target individual.

In contrast, the [open-population identification](https://github.com/kabuga1987/Cricket_Identification/tree/main/open_population_ID) assumes that the number of individuals is unknown and may change over time. In this setting, the classifier takes a pair of spectrograms for CNNs or a pair of MFCC feature vectors for RFs and predicts a similarity score indicating whether the two samples originate from the same individual or from different individuals.

To benchmark the performance of models trained from scratch, transfer learning using a pre-trained model was applied in both the [closed-](https://github.com/kabuga1987/Cricket_Identification/tree/main/pretraining_closed_population_ID) and [open-population](https://github.com/kabuga1987/Cricket_Identification/tree/main/pretraining_open_population_ID) settings.

This code accompanies the paper: **Acoustic individual identification in a species of field cricket using deep learning**


## Methodology overview
The summary of our methodologies is illustrated below.

<p align="center">
<img src="images/summary.pdf" width="700">
</p>

## Preprocessing

Before acoustic individual identification, two acoustic-based segmentation approaches were implemented to divide raw cricket recordings into meaningful acoustic units. A **threshold-based chirp detector** and fixed-length segment partitioning[segment_index_extraction](https://github.com/kabuga1987/Cricket_Identification/tree/main/segment_index_extraction). The [chirp detector](https://github.com/kabuga1987/Cricket_Identification/tree/main/chirp_index_extraction) identifies chirp start and end times as the earliest and latest times exceeding a signal-to-noise ratio amplitude of 0.1, with a maximum of 45 ms between threshold crossings for points to be in the same chirp. The figure below demonstrates this process, with vertical lines indicating the start times and green end times.

![chirp index](chirp_index_extractor/chirp.pdf)

1-second fixed-length partition approach....



This project integrates preprocessing, feature extraction, and identification steps to ascertain the feasibility of identifying cricket individuals exclusively from their calls.

1. The preprocessing step ([chirp_index_extraction](https://github.com/kabuga1987/Cricket_Identification/tree/main/chirp_index_extraction) or [segment_index_extraction](https://github.com/kabuga1987/Cricket_Identification/tree/main/segment_index_extraction)) applies a noise reduction algorithm and a lower filter to remove noise before segmenting the cricket signal into acoustically relevant units.
2. The [feature extraction](https://github.com/kabuga1987/Cricket_Identification/tree/main/segment_spectrogram_mfcc_feature_extraction) step generates spectrograms, MFCCs, and temporal features from the segmented acoustic units.
3. The identification step utilizes standard classification models, such as Random Forests (RFs) and Convolutional Neural Networks (CNNs), to address cricket individual identification in a [closed population setting](https://github.com/kabuga1987/Cricket_Identification/tree/main/closed_population_ID), where the number of calling individuals is known. In this case, the task is to assign a call to one of the existing individuals. The models process input data (e.g., MFCC vectors for RFs or spectrograms for CNNs) to predict the associated individual categories. In contrast, for [open population identification](https://github.com/kabuga1987/Cricket_Identification/tree/main/open_population_ID), where the number of individuals is unknown and may vary over time, similarity learning models are employed. These models work by processing input pairs (e.g., pairs of MFCCs for RFs or spectrograms for CNNs) to determine whether the calls were made by the same individual or different individuals.



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



