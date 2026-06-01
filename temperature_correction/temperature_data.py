import os
import pandas as pd
import numpy as np
#======================

class TemperatureData:

    def __init__(self, path_temp_data, path_chirp_data):
        self.path_temp_data = path_temp_data
        self.path_chirp_data = path_chirp_data

    def audio_temp_df(self):
        audio_temp_df = pd.read_excel(self.path_temp_data)
        return audio_temp_df

    def audio_to_temperature(self):
        odio_temp_df = self.audio_temp_df()
        audio_2_temp = {audio + ".WAV":temp for audio, temp in odio_temp_df.to_records(index = False)}
        return audio_2_temp  

    def chirp_to_audio(self):
        df = pd.read_csv(self.path_chirp_data)
        df_ch_odio = df[['chirp','audio']]
        chirp_2_audio = {chirp:audio for chirp, audio in df_ch_odio.to_records(index = False)}
        audio_2_temp = self.audio_to_temperature()
        chirps = df["chirp"].values
        temp = [audio_2_temp[chirp_2_audio[chirps[i]]] for i in range(len(chirps))]
        df["temp"] = temp
        df["ID"] = df["chirp"].str[:3]
        dk = df[["chirp","ID","chDur","carrier_freq","temp"]]
        return  dk

#=====================

if __name__ == "__main__":
    p_features = "../5syllable_chirp_spectrogram_creation/mfcc_tfs_carrier_freq_5syllable_chirps.csv"
    p_temp = "recording_temperature_data.xlsx"
    td = TemperatureData(p_temp, p_features)
    df = td.chirp_to_audio()
    df.to_csv("temperature_data.csv", index = False)
    print()
    print()
    print(f"View the created data frame")
    print(df)