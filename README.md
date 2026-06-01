# Cricket individual identification
## Introduction
This project integrates preprocessing, feature extraction, and identification steps to assess the feasibility of identifying cricket individuals exclusively from their acoustic recordings using deep learning and machine learning models in both closed- and open-population settings through different temporal scales.

This code accompanies the paper: **Acoustic individual identification in a species of field cricket using deep learning**

## Authors
Emmanuel Kabuga, Diptarup Nandi, Stuart Burrell, Gciniwe Dlamini, Rohini Balakrishnan, Bubacarr Bah, Ian Durbach

## Requirements


All required packages can be installed  using `pip install -r requirements.txt`

Numpy, Pandas, Sklearn, Tensorflow, Keras, Librosa, and Scipy




This project develops deep learning and random forest classifiers for identifying individual crickets in both closed- and open-population settings.

In the [closed-population identification](https://github.com/kabuga1987/Cricket_Identification/tree/main/closed_population_ID), the number of individuals is assumed to be known and fixed throughout the study period. Under this setting, the classifier takes a spectrogram as input for convolutional neural networks (CNNs) or an MFCC feature vector for random forests (RFs) and predicts the probability that the sample belongs to each target individual.

In contrast, the [open-population identification](https://github.com/kabuga1987/Cricket_Identification/tree/main/open_population_ID) assumes that the number of individuals is unknown and may change over time. In this setting, the classifier takes a pair of spectrograms for CNNs or a pair of MFCC feature vectors for RFs and predicts a similarity score indicating whether the two samples originate from the same individual or from different individuals.

To benchmark the performance of models trained from scratch, transfer learning using a pre-trained model was applied in both the [closed-](https://github.com/kabuga1987/Cricket_Identification/tree/main/pretraining_closed_population_ID) and [open-population](https://github.com/kabuga1987/Cricket_Identification/tree/main/pretraining_open_population_ID) settings.




## Methodology overview
The summary of our methodologies is illustrated below.

<p align="center">
<img src="images/summary.pdf" width="700">
</p>

## Preprocessing

Before acoustic individual identification, two acoustic-based segmentation approaches were implemented to divide raw cricket recordings into meaningful acoustic units: a **threshold-based chirp detector** and **fixed-length signal segmentation** 

### Threshold-based chirp detector



 The [chirp detector](https://github.com/kabuga1987/Cricket_Identification/tree/main/chirp_index_extraction) identifies chirp start and end times as the earliest and latest points at which the signal-to-noise ratio (SNR) amplitude exceeds a threshold of 0.1. Consecutive threshold crossings are considered part of the same chirp if they are separated by no more than 45 ms (2,000 samples). In the first stage, the algorithm processes an audio file and outputs the start and end times of each detected chirp. An example is shown in the figure below, where the red vertical lines indicate chirp start times and the green vertical lines indicate chirp end times. For implementation details, see [`chirp_index_extractor.ipynb`](https://github.com/kabuga1987/Cricket_Identification/blob/main/chirp_index_extraction/chirp_index_extractor.ipynb).

 
<p align="center">
<img src="https://github.com/kabuga1987/Cricket_Identification/blob/main/chirp_index_extraction/chirp.png" width="700">
</p>



In the second stage, the algorithm is used to detect syllables within each extracted chirp, enabling the subsequent computation of temporal features such as syllable duration and inter-syllable interval. For syllable detection, a maximum gap of 5.3 ms (235 samples) between threshold crossings is allowed for points to be considered part of the same syllable. At this stage, the algorithm takes a chirp signal as input and outputs the start and end times of each detected syllable. These outputs are then used to calculate the number of syllables per chirp and other temporal characteristics. The figure below illustrates the syllable detection process.

![chirp index](chirp_index_extractor/syllable.pdf)

### One-second segment extraction

[Fixed-length segmenatation approach ](https://github.com/kabuga1987/Cricket_Identification/tree/main/segment_index_extraction) divides each audio recording into segments of 1-second duration, with 80% overlap between successive segments. The figure below displays an example of 1-second segment waveform.

## Acoustic features

This project exploited three types of acoustic features to identify cricket individuals: spectrograms, MFCCs, and temporal features. Spectrograms and MFCC vectors were created from both 1-second and five-syllable segments, while temporal features were only extracted from five-syllable chirps.







