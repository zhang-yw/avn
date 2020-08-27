import numpy as np
import librosa
import os
from keras.utils import to_categorical

def wav2mfcc(file_path, max_pad_len=128):
    wave, sr = librosa.load(file_path, mono=False, sr=None)
    wave_1 = wave[0][::2]
    mfcc_1 = librosa.feature.melspectrogram(wave_1, sr=sr)
    pad_width = max_pad_len - mfcc_1.shape[1]
    mfcc_1 = np.pad(mfcc_1, pad_width=((0, 0), (0, pad_width)), mode='constant')
    wave_2 = wave[1][::2]
    mfcc_2 = librosa.feature.melspectrogram(wave_2, sr=sr)
    pad_width = max_pad_len - mfcc_2.shape[1]
    mfcc_2 = np.pad(mfcc_2, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return np.stack([mfcc_1, mfcc_2], axis=2)
