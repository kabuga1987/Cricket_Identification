import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt
#===============================
class SegmentStartEndPointExtractor:
    """
    Extract fixed-length segment start and end points
    from audio recordings.
    """

    def __init__(
        self,
        path_audio_files,
        path_save_segment_data,
        start=None,
        sr=None,
        window=None,
        slide=None
    ):
        """
        Parameters
        ----------
        path_audio_files : str
            Directory containing audio files.

        path_save_segment_data : str
            Path where the CSV file will be saved.

        start : int or float
            Time (in seconds) from which segmentation starts.

        sr : int or None
            Sampling rate. If None, original sampling rate is used.

        window : int or float
            Segment duration in seconds.

        slide : int or float
            Sliding step between consecutive segments in seconds.
        """

        self.path_audio_files = path_audio_files
        self.path_save_segment_data = path_save_segment_data
        self.start = start
        self.sr = sr
        self.window = window
        self.slide = slide

    def Audio_reader(self, file_name):
        """
        Read an audio file.
        
        Parameters
        ----------
        file_name : str
            Audio filename.

        Returns
        -------
        tuple
            signal, sampling_rate
        """
        
        full_path = os.path.join(self.path_audio_files,file_name)
        signal, sr = librosa.load(full_path,sr=self.sr)
        return signal, sr

    def Start_End_locations(self, audio_signal, sr):
        """
        Extract segment start and end sample indices.

        Parameters
        ----------
        audio_signal : np.ndarray
            Audio signal.

        sr : int
            Sampling rate.

        Returns
        -------
        tuple
            start_indices, end_indices
        """

        st_ed_locations = []
        start_sample = int(self.start * sr)
        window_samples = int(self.window * sr)
        slide_samples = int(self.slide * sr)
        current_start = start_sample

        while True:
            current_end = current_start + window_samples
            if current_end > len(audio_signal):
                break
            st_ed_locations.append((current_start, current_end))
            current_start += slide_samples
        if len(st_ed_locations) == 0:
            return np.array([]), np.array([])
        starts, ends = zip(*st_ed_locations)

        return np.array(starts), np.array(ends)

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
            raise ValueError(f"Unexpected filename format: {audiofile}")
        audio_id = f"{ID}{night}{call}"
        return audio_id, ID, int(night), int(call)

    @staticmethod
    def Data_Frame():
        """
        Create empty segment dataframe.
        """
        columns = [
            "ID",
            "Start",
            "End",
            "Night",
            "Call",
            "Audio"
        ]
        return pd.DataFrame(columns=columns)
        
    def plot_figure(self, xx, start,end):
        _1s_signal = xx[start:end]
        plt.figure(figsize = (5,3))
        plt.plot(_1s_signal)
        plt.savefig("1s_signal.png")
        plt.show()

    def Execution(self):
        """
        Run segmentation on all audio files.

        Returns
        -------
        pd.DataFrame
            Dataframe containing segment metadata.
        """
        all_rows = []
        for audio_file in tqdm(os.listdir(self.path_audio_files)):
            if not audio_file.lower().endswith(
                (".wav", ".mp3", ".flac")
            ):
                continue
            audio_signal, sr = self.Audio_reader(audio_file)
            starts, ends = self.Start_End_locations(
                audio_signal,
                sr
            )
            
            if len(starts) == 0:
                continue
            audio_id, ID, night, call = (self.ID_night_call(audio_file))
            for idx, (st_idx, ed_idx) in enumerate(zip(starts, ends)):
                segment_name = f"{audio_id}_{idx}"

                all_rows.append({
                    "Segment": segment_name,
                    "ID": ID,
                    "Start": int(st_idx),
                    "End": int(ed_idx),
                    "Night": night,
                    "Call": call,
                    "Audio": audio_file
                })

        df = pd.DataFrame(all_rows)
        if not df.empty:
            df = df.set_index("Segment")
        # Save extracted segment metadata
        df.to_csv(self.path_save_segment_data,index=True)

        return df
#===================================

if __name__ == "__main__":
    pathAudioFiles = "../audio_files/"
    pathSegData = "segment_data.csv"
    start = 10 # starting analysis 10 sec after the initiation of the call
    sr = 44100 # sample rate
    window = 1 # frame, window, or segment size (1s)
    slide = 0.2 # time difference btwn two consecutive segments (0.2s)
    sE = SegmentStartEndPointExtractor(pathAudioFiles,pathSegData,start=start,sr=sr, window=window,slide=slide)
    df = sE.Execution()
    print()
    print()
    print(f"View the created data frame")
    print()
    print(df)
    print()
    print(f"View and example of 1-second segment waveform")
    signal1 = librosa.load(pathAudioFiles + "RBB_n3_1.WAV", sr = None, offset=None)[0]
    signal2 = librosa.load(pathAudioFiles + "RBB_n3_1.WAV", sr = None, offset=None)[0]
    sE.plot_figure(signal1,3580920,3625020)
    