## Cricket Individual animal identification
This project integrates preprocessing, feature extraction, and identification steps to ascertain the feasibility of identifying cricket individuals from exclusively their calls.

1. The preprocessing step applies a noise reduction algorithm and a lower filter to remove noise before segmenting the cricket signal into acoustically relevant units.
2. The feature extraction step generates spectrograms, MFCCs, and temporal features from the produced acoustic units.
3. The identification step utilizes standard classification models, such as Random Forests (RFs) and Convolutional Neural Networks (CNNs), to address cricket individual identification in a closed population setting, where the number of calling individuals is known. In this case, the task is to assign a call to one of the existing individuals. The models process input data (e.g., MFCC vectors for RFs or spectrograms for CNNs) to predict the associated individual categories. In contrast, for open population identification, where the number of individuals is unknown and may vary over time, similarity learning models are employed. These models work by processing input pairs (e.g., pairs of MFCCs for RFs or spectrograms for CNNs) to determine whether the calls were made by the same individual or different individuals.

This code accompanies the paper: **Individual identification of field crickets from their calls using similarity comparison networks**

## Authors
Emmanuel Kabuga, Diptarup Nandi, Stuart Burrell, Gciniwe Dlamini, Bubacarr Bah, Ian Durbach

## Requirements


All required packages can be installed  using `pip install -r requirements.txt`

Numpy

Pandas

Scipy

Sklearn

Tensorflow

Keras





