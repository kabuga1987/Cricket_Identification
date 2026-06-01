import os
import numpy as np
import pandas as pd
import librosa
import noisereduce as nr
from scipy import signal
from tqdm import tqdm
import skimage
import matplotlib.pyplot as plt
#=====================
class SpectrogramAndMFCCExtraction:
    """
    Extract mel spectrograms and MFCC features
    from segmented audio recordings.
    """

    def __init__(
        self,
        path_audio_files,
        path_segment_data,
        path_save_spec_arrays,
        cutoff,
        nyq_fr,
        **kwargs
    ):
        """
        Parameters
        ----------
        path_audio_files : str
            Directory containing audio files.

        path_segment_data : str
            CSV file containing segment annotations.

        path_save_spec_arrays : str
            Directory where spectrogram arrays will be saved.

        cutoff : float
            Low-pass filter cutoff frequency.

        nyq_fr : float
            Nyquist frequency.
        """

        self.path_audio_files = path_audio_files
        self.path_segment_data = path_segment_data
        self.path_save_spec_arrays = path_save_spec_arrays
        self.cutoff = cutoff
        self.nyq_fr = nyq_fr
        self.kwargs = kwargs

        # Create output directory if it does not exist
        os.makedirs(
            self.path_save_spec_arrays,
            exist_ok=True
        )

    def read_audio_file(self, audio_file):
        """
        Read an audio file.
        Parameters
        ----------
        audio_file : str, Audio filename.
        Returns
        -------
        np.ndarray, Audio signal.
        """
        full_path = os.path.join(self.path_audio_files,audio_file)
        signal_data, _ = librosa.load(full_path,sr=self.kwargs["sr"])
        return signal_data

    def reduce_noise(self, signal_data):
        """
        Reduce background noise.
        Parameters
        ----------
        signal_data : np.ndarray, Audio signal.
        Returns
        -------
        np.ndarray, Noise-reduced signal.
        """
        reduced_noise = nr.reduce_noise(y=signal_data,sr=self.kwargs["sr"])
        return reduced_noise

    @staticmethod
    def butter_lowpass(cutoff, nyq_freq, order=4):
        """
        Create Butterworth low-pass filter coefficients.
        """
        normal_cutoff = float(cutoff) / nyq_freq
        b, a = signal.butter(order, normal_cutoff, btype="lowpass")
        return b, a

    def butter_lowpass_filter(self, data, cutoff_freq, nyq_freq, order=4):
        """
        Apply a low-pass Butterworth filter.
        Parameters
        ----------
        data : np.ndarray, Audio signal.
        cutoff_freq : float, Cutoff frequency.
        nyq_freq : float, Nyquist frequency.
        order : int, Filter order.
        Returns
        -------
        np.ndarray, Filtered signal.
        """
        b, a = self.butter_lowpass(cutoff_freq, nyq_freq, order=order)
        filtered_signal = signal.filtfilt(b, a, data)
        return filtered_signal

    def preprocess_audio_file(self, audio_file):
        """
        Complete preprocessing pipeline.
        Steps
        -----
        1. Read audio
        2. Reduce noise
        3. Apply low-pass filter
        Returns
        -------
        np.ndarray
            Preprocessed signal.
        """
        signal_data = self.read_audio_file(audio_file)
        reduced_noise = self.reduce_noise(signal_data)
        filtered_signal = self.butter_lowpass_filter(
            reduced_noise,
            self.cutoff,
            self.nyq_fr
        )
        return filtered_signal

    def read_segment_data(self):
        """
        Read segment annotation CSV.
        Returns
        -------
        list
            List of tuples:
            (audio, segment, start, end)
        """
        df = pd.read_csv(self.path_segment_data)
        required_columns = [
            "Segment",
            "Start",
            "End",
            "Audio"
        ]
        missing_cols = [
            col for col in required_columns
            if col not in df.columns
        ]
        if missing_cols:
            raise ValueError(
                f"Missing columns: {missing_cols}"
            )
        audio_segment_st_end = list(
            zip(
                df["Audio"],
                df["Segment"],
                df["Start"],
                df["End"]
            )
        )

        return audio_segment_st_end

    def audio2_segment_indexes(self):
        """
        Map audio files to their segments.
        Returns
        -------
        dict
            {
                audio_file:
                [(segment, start, end), ...]
            }
        """
        audio_segment_st_end = (self.read_segment_data())
        audio2data = {}
        for audio, segment, st, end in (audio_segment_st_end):
            if audio not in audio2data:
                audio2data[audio] = []
            audio2data[audio].append((segment, int(st), int(end)))
        return audio2data

    @staticmethod
    def audio_frame_to_mel_spectrogram(audio_frame, **kwargs):
        """
        Extract mel spectrogram.
        Parameters
        ----------
        audio_frame : np.ndarray, Audio segment.
        Returns
        -------
        np.ndarray, Mel spectrogram.
        """

        mel_spectrogram = (
            librosa.feature.melspectrogram(
                y=audio_frame.astype(np.float32),
                **kwargs
            )
        )

        return mel_spectrogram

    def segment_mfccs(self, segment_signal):
        """
        Extract MFCC statistics.
        Parameters
        ----------
        segment_signal : np.ndarray, Audio segment.
        Returns
        -------
        np.ndarray, Concatenated MFCC means and standard deviations.
        """
        mfccs = librosa.feature.mfcc(
            y=segment_signal,
            sr=self.kwargs["sr"],
            n_mfcc=20,
            n_mels=128,
            fmin=4000,
            fmax=8000,
            n_fft=1024,
            hop_length=256
        )

        means = mfccs.mean(axis=1)
        stdvs = mfccs.std(axis=1)
        features = np.hstack([means, stdvs])
        return features

    def Execution(self):
        """
        Run spectrogram and MFCC extraction.
        Parameters
        ----------
        **kwargs :
            Additional parameters for
            librosa.feature.melspectrogram()
        Returns
        -------
        pd.DataFrame, DataFrame containing MFCC features.
        """
        audio2data = self.audio2_segment_indexes()
        segment_2_mfccs = {}
        for audio, data in tqdm(audio2data.items()):
            processed_signal = (
                self.preprocess_audio_file(audio)
            )
            for segment_data in data:
                name, st, end = segment_data
                segment_signal = processed_signal[st:end]
                if len(segment_signal) == 0:
                    continue
                # Mel spectrogram
                mel_spec = (
                    self.audio_frame_to_mel_spectrogram(
                        segment_signal,
                        **self.kwargs
                    )
                )
                
                spec_path = os.path.join(self.path_save_spec_arrays,f"{name}.npy")
                np.save(spec_path, mel_spec)
                # MFCC extraction
                mfccs = self.segment_mfccs(segment_signal)
                segment_2_mfccs[name] = mfccs

        mfcc_names = np.array([f"x{i}" for i in range(40)])
        df = pd.DataFrame.from_dict(
            segment_2_mfccs,
            orient="index",
            columns=mfcc_names
        )
        df.index.name = "Segment"
        df.to_csv(
            "mfcc_data.csv",
            index=True
        )
        return df
#======================================

if __name__ == "__main__":
    p_odio = "../audio_files/"
    p_anno = "../segment_index_extraction/segment_data.csv"
    p_arra = "spectrogram_arrays/"
    cutoff = 20000
    nyq_fr = 22050
    WINDOW_LENGTH= 1024 # size of the window when applying STFT
    HOPE_LENGTH = 256   # number of samples to jump between two consecutive windows
    NUMBER_MELS = 128  # mels bins, this will be the HEIGHT of the mel spectrogram
    SAMPLE_RATE = 44100 # sample rate used to down-sample the audio file
    POWER = 1            # power of order 1, order i.e. squared energy
    F_MIN = 4000       # minimum frequency to be detected
    F_MAX  = 8000 
    PARAMETERS = {"n_fft":WINDOW_LENGTH,
              "hop_length":HOPE_LENGTH,
              "n_mels":NUMBER_MELS,
              "sr":SAMPLE_RATE,
              "power":POWER,
              "fmin":F_MIN,
              "fmax":F_MAX
             }
    cricket = SpectrogramAndMFCCExtraction(p_odio, p_anno, p_arra, cutoff, nyq_fr, **PARAMETERS)
    df = cricket.Execution()
    
    print()
    print()
    print(f"View the created data frame")
    print()
    print()
    print(df)

    print()
    print()
    print(f"View a 1-second example spectrogram")
    print()
    print()
    imgs = os.listdir(p_arra)
    img = np.load(p_arra+imgs[2])
    plt.imshow(img)
    plt.axis("off")
    plt.show()
        