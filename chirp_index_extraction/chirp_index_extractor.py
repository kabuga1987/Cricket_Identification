import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
#==================================

class ChirpStartEndPointExtractor:
    """
    Extract cricket chirp start/end points and associated metadata
    from audio recordings. The analysis starts after 10s listing the
    identifier code of the recorded individual.
    """

    def __init__(
        self,
        path_to_audio,
        spacing=1000,
        padding=100,
        amplitude=0.1,
        start=10
    ):
        """
        Parameters
        ----------
        path_to_audio : str
            Directory containing audio files.

        spacing : int
            Minimum distance (in samples) separating two chirps.

        padding : int
            Number of samples added before and after chirp boundaries.

        amplitude : float
            Amplitude threshold used to detect chirp activity.

        start : int or float
            Number of seconds to skip at the beginning of the recording.
        """

        self.path_to_audio = path_to_audio
        self.spacing = spacing
        self.padding = padding
        self.amplitude = amplitude
        self.start = start

    def read_normalize_cricket_signal(self, audio_file):
        """
        Read and normalize an audio signal.

        Parameters
        ----------
        audio_file : str
            Audio filename.

        Returns
        -------
        np.ndarray
            Normalized audio signal.
        """

        full_path = os.path.join(self.path_to_audio, audio_file)
        signal, sr = librosa.load(full_path,sr=None)
        # Remove first `start` seconds
        signal = signal[int(sr * self.start):]
        max_val = np.max(np.abs(signal))
        if max_val == 0:
            return signal
        return signal / max_val

    def detect_cricket_call(self, normalized_signal, spacing=None, padding=None, amplitude=None):
        """
        Detect chirp start and end indices.

        Parameters
        ----------
        normalized_signal : np.ndarray
            Normalized audio signal.

        spacing : int, optional
            Minimum separation between chirps.

        padding : int, optional
            Padding added around chirp boundaries.

        amplitude : float, optional
            Detection threshold.

        Returns
        -------
        tuple
            start_call_padded, end_call_padded
        """

        spacing = spacing or self.spacing
        padding = padding or self.padding
        amplitude = amplitude or self.amplitude

        # Detect samples exceeding threshold
        active_samples = np.where(np.abs(normalized_signal) > amplitude)[0]
        if active_samples.size == 0:
            return np.array([]), np.array([])
        # Difference between consecutive active samples
        gaps = np.diff(active_samples)
        # Add large artificial gap at beginning
        gaps = np.insert(gaps, 0, spacing * 5)
        # Chirp starts
        start_call = active_samples[gaps > spacing]
        # Chirp ends
        inverted_signal = normalized_signal[::-1]
        active_samples_inv = np.where(np.abs(inverted_signal) > amplitude)[0]
        gaps_inv = np.diff(active_samples_inv)
        gaps_inv = np.insert(gaps_inv, 0, spacing * 5)
        end_call = (len(normalized_signal)- active_samples_inv[gaps_inv > spacing])
        end_call = end_call[::-1]

        # Padding
        start_call_padded = np.maximum(0,start_call - padding)
        end_call_padded = np.minimum(len(normalized_signal),end_call + padding)
        # Ensure equal lengths
        min_len = min(len(start_call_padded), len(end_call_padded))

        return (start_call_padded[:min_len],end_call_padded[:min_len])

    def check_click_detector(self, signal, click_start, click_end):
        """
        Visualize detected chirp boundaries.
        """
        plt.figure(figsize=(10, 4))
        plt.plot(signal)
        plt.vlines(
            x=click_start,
            ymin=signal.min(),
            ymax=signal.max(),
            color="red"
        )

        plt.vlines(
            x=click_end,
            ymin=signal.min(),
            ymax=signal.max(),
            color="green"
        )
        plt.savefig("chirp.png")

        plt.show()

    def ID_night_call(self, audiofile):
        """
        Extract ID, night, and call number from filename.

        Expected formats
        ----------------
        XXX_nN_P.WAV
        XXX_P.WAV

        Returns
        -------
        tuple
            audio_id, ID, night, call
        """

        audio = os.path.splitext(audiofile)[0]
        if len(audio) == 8:
            ID = audio[:3]
            night = audio[-1]
            call = audio[-3]
        elif len(audio) == 5:
            ID = audio[:3]
            night = audio[-1]
            call = "1"
        else:
            raise ValueError(
                f"Unexpected filename format: {audiofile}"
            )

        audio_id = f"{ID}{night}{call}"

        return audio_id, ID, int(night), int(call)

    @staticmethod
    def Data_Frame():
        """
        Create empty chirp dataframe.
        """

        columns = [
            "ID",
            "Start",
            "End",
            "Night",
            "Call",
            "Syllables",
            "Audio"
        ]

        return pd.DataFrame(columns=columns)

    def Execution(self, space, pad, amp):
        """
        Run chirp extraction over all audio files.

        Parameters
        ----------
        space : int
            Spacing parameter for syllable detection.

        pad : int
            Padding parameter for syllable detection.

        amp : float
            Amplitude threshold for syllable detection.

        Returns
        -------
        pd.DataFrame
            Dataframe containing chirp information.
        """

        all_rows = []
        for audiofile in os.listdir(self.path_to_audio):
            if not audiofile.lower().endswith((".wav", ".mp3", ".flac")):
                continue
            normalized_signal = (self.read_normalize_cricket_signal(audiofile))
            start_, end_ = self.detect_cricket_call(normalized_signal, self.spacing, self.padding, self.amplitude)
            # Ignore first chirp if required
            start = start_[1:]
            end = end_[1:]
            if len(start) == 0:
                continue
            audio_id, ID, night, call = (self.ID_night_call(audiofile))
            for idx, (st, ed) in enumerate(zip(start, end)):
                chirp_signal = normalized_signal[st:ed]
                st_syl, end_syl = self.detect_cricket_call(chirp_signal, space, pad, amp)

                nb_syllables = len(end_syl)
                chirp_name = f"{audio_id}_{idx}"
                all_rows.append({
                    "Chirp": chirp_name,
                    "ID": ID,
                    "Start": int(st),
                    "End": int(ed),
                    "Night": night,
                    "Call": call,
                    "Syllables": nb_syllables,
                    "Audio": audiofile
                })

        df = pd.DataFrame(all_rows)
        if not df.empty:
            df = df.set_index("Chirp")
        df.to_csv("chirp_data.csv", index=True)

        return df

#======================

if __name__ == "__main__":
    pathAudioFiles = "../audio_files/"
    chirp = ChirpStartEndPointExtractor(pathAudioFiles, 2000, 200,0.1,10 )
    df = chirp.Execution(235, 200,0.1)
    print()
    print()
    print(f"View the created data frame")
    print(df)
    print
    print(f"View an example of the created spectrograms")
    print()
    print()
    signal1 = librosa.load(pathAudioFiles + "BKB_n2_1.WAV", sr = None, offset=10)[0]
    plt.plot(signal1[18826:26083])
    plt.show
    print()
    print()
    print(f"Illustration of the chirp detection algorithm")
    print()
    st = 444200
    end = 494200
    # Detecting chirps in the portion of the signal1 starting from st up to end
    clickstart, clickend = chirp.detect_cricket_call(signal1[st:end], 2000, 200, 0.1)
    chirp.check_click_detector(signal1[st:end],clickstart, clickend)
    print()