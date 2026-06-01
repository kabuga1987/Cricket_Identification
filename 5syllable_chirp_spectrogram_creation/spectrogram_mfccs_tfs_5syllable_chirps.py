import os
import numpy as np
import pandas as pd
import librosa
import noisereduce as nr
from scipy import signal
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
#===============================

class SpecMFCCTemporalFeatures:

    def __init__(self, path_csv, path_odio, path_save, start, syllabs, cutoff, nyq_freq, padding, **kwargs):
        self.path_csv = path_csv
        self.path_odio = path_odio
        self.path_save = path_save
        self.start = start
        self.syllabs = syllabs
        self.cutoff = cutoff
        self.nyq_freq = nyq_freq
        self.padding = padding
        self.kwargs = kwargs

        self.df = pd.read_csv(path_csv)
        self.df = self.df[self.df.Syllables == syllabs]

        # Precompute filter coefficients once (optimization)
        self.b, self.a = signal.butter(4, cutoff / nyq_freq, btype="lowpass")

    def chirp_2_idx(self):
        """Map chirp to (start, end) indices"""
        return {chirp: (st, ed) for chirp, _, st, ed, _, _, _, _ in self.df.to_records(index=False)}

    def chirp_2_audio(self):
        """Map chirp to audio file"""
        return {chirp: audio for chirp, _, _, _, _, _, _, audio in self.df.to_records(index=False)}

    @staticmethod
    def audio_2_chirps(ch2odio_dict):
        """Group chirps by audio file"""
        audio2ch = {}
        for chirp, audio in ch2odio_dict.items():
            if audio not in audio2ch:
                audio2ch[audio] = []
            audio2ch[audio].append(chirp)
        return audio2ch

    def butter_lowpass_filter(self, data):
        """Apply lowpass filter using precomputed coefficients"""
        return signal.filtfilt(self.b, self.a, data)

    def call_signal(self, audiofile):
        """Load audio, reduce noise, and apply filtering"""
        signal_raw, sr = librosa.load(os.path.join(self.path_odio, audiofile), sr=None, offset=self.start)
        reduced_noise = nr.reduce_noise(y=signal_raw, sr=sr)
        filtered = self.butter_lowpass_filter(reduced_noise)
        filtered = filtered / (np.abs(filtered).max() + 1e-12)
        return filtered, sr

    @staticmethod
    def detect_cricket_call(signal, spacing=235, padding=50):
        """Detect syllable start and end positions"""
        idx = np.where(np.abs(signal) > 0.1)[0]
        
        if len(idx) == 0:
            return np.array([]), np.array([])
            
        diff = np.diff(idx)
        starts = idx[np.hstack(([True], diff > spacing))]

        rev_signal = signal[::-1]
        idx_rev = np.where(np.abs(rev_signal) > 0.1)[0]
        diff_rev = np.diff(idx_rev)

        ends = rev_signal.shape[0] - idx_rev[np.hstack(([True], diff_rev > spacing))]

        starts = np.maximum(0, starts - padding)
        ends = np.minimum(ends[::-1] + padding, signal.shape[0])

        return starts, ends

    
    def carrier_frequency(self, signal, sr):
        """Estimate dominant carrier frequency"""
        D = librosa.stft(signal)
        freqs = librosa.fft_frequencies(sr=sr)
        return freqs[np.argmax(np.mean(np.abs(D), axis=1))]

    
    def mfcc_features(self, signal):
        """Extract MFCC mean and std features"""

        mfccs = librosa.feature.mfcc(
            y=signal, sr=self.kwargs["sr"],
            n_mfcc=20, n_mels=self.kwargs["mels"],
            fmin=self.kwargs["min"], fmax=self.kwargs["max"],
            n_fft=self.kwargs["fft"], hop_length=self.kwargs["hop"]
        )

        return np.hstack([mfccs.mean(axis=1), mfccs.std(axis=1)])


    def spectrogram_features(self, signal):     
        
        spectrogram = librosa.feature.melspectrogram(
            y=signal, 
            n_fft = self.kwargs["fft"],
            hop_length= self.kwargs["hop"],
            n_mels = self.kwargs["mels"],
            sr = self.kwargs["sr"],
            power = self.kwargs["power"],
            fmin = self.kwargs["min"],
            fmax = self.kwargs["max"]      
        )
        return spectrogram

    def pad(self, signal,max = 10687):
        
        length = len(signal)
        if length < max:
            dx = max - length
            pad_left = dx//2
            pad_right = dx-pad_left
            padded = np.pad(signal,(pad_left, pad_right), mode = "constant", constant_values=0)
        else:
            padded = signal
        return padded

    def save_feature(self, feature, file_name):
        """Save feature array."""
        save_file = os.path.join(self.path_save, file_name + ".npy")
        np.save(save_file, feature)
         
    def temporal_feature_names(self):
        """Generate feature column names"""

        if self.syllabs == 5:
            tfs = ['chDur','s1Dur','s2Dur','s3Dur','s4Dur','s5Dur','meanSyl',
                   's12Gap','s23Gap','s34Gap','s45Gap','meanGap','carrier_freq']
        elif self.syllabs == 4:
            tfs = ['chDur','s1Dur','s2Dur','s3Dur','s4Dur','meanSyl','s12Gap',
                   's23Gap','s34Gap','meanGap','carrier_freq']
        elif self.syllabs == 3:
            tfs = ['chDur','s1Dur','s2Dur','s3Dur','meanSyl','s12Gap','s23Gap',
                   'meanGap','carrier_freq']
        else:
            tfs = ['chDur','s1Dur','s2Dur','meanSyl','s12Gap','meanGap',
                   'carrier_freq']

        mfcc_names = [f"x{i}" for i in range(40)]
        return ["audio"]+ mfcc_names + tfs

    def Execution(self):
        """Main feature extraction pipeline"""

        ch2idx = self.chirp_2_idx()
        ch2odio = self.chirp_2_audio()
        audio2chirps = self.audio_2_chirps(ch2odio)

        chfs = {}

        for audio, chirps in tqdm(audio2chirps.items()):
            signal, sr = self.call_signal(audio)
            for chirp in chirps:
                st, ed = ch2idx[chirp]
                chirp_sig = signal[st:ed]
                starts, ends = self.detect_cricket_call(chirp_sig)
                if ends.shape[0] != self.syllabs:
                    continue

                chDurs = np.hstack([(ed-self.padding)-(st+self.padding), (ends-starts), (ends-starts).mean()])
                chGaps = np.hstack([(starts[1:]-ends[:-1]), (starts[1:]-ends[:-1]).mean()])

                mfcc_feats = self.mfcc_features(chirp_sig)
                carrier_freq = self.carrier_frequency(chirp_sig, sr)

                chfs[chirp] = np.hstack([audio, mfcc_feats, chDurs, chGaps, carrier_freq])

                # pad to chirp to the maximum chirp to have same resolution for all chirps
                padded = self.pad(chirp_sig)
                spec = self.spectrogram_features(padded)
                self.save_feature(spec, chirp)

        df = pd.DataFrame.from_dict(chfs, orient='index', columns=self.temporal_feature_names())
        return df.rename_axis("chirp")

#=====================================

if __name__ == "__main__":

    path_chirp_data = "../chirp_index_extraction/chirp_data.csv"
    path_odio = "../audio_files/"
    path_ars = "5syl_spec_arrays/"
    start = 10 # remove the first 10s listing the cricked identifier
    syllabs = 5
    cutoff = 8000 
    nyq_freq = 44100 / 2 # half of the sample rate
    padding = 400
    
    kwargs = {"fft": 1024,
            "hop": 256,
            "mels": 128,
            "sr": 44100,
            "power": 1,
            "min": 4000,
            "max": 8000    
}

    tF = SpecMFCCTemporalFeatures(path_chirp_data, path_odio, path_ars, start, syllabs, cutoff, nyq_freq, padding, **kwargs)
    df = tF.Execution()
    df.to_csv("mfcc_tfs_carrier_freq_5syllable_chirps.csv", index = True)
    print()
    print()
    print(f"View the created data frame")
    print()
    print()
    print(f"View an example of the created spectrograms")
    print()
    specs = os.listdir(path_ars)
    spec = np.load(path_ars + specs[1])
    plt.imshow(spec)
    plt.axis("off")
    plt.show()