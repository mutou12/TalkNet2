from scipy.io.wavfile import read
import numpy as np
import librosa
import pyworld as pw


def read_wav_np(path, sample_rate):
    sr, wav = read(path)
    if sr == sample_rate:
        if len(wav.shape) == 2:
            wav = wav[:, 0]

        if wav.dtype == np.int16:
            wav = wav / 32768.0
        elif wav.dtype == np.int32:
            wav = wav / 2147483648.0
        elif wav.dtype == np.uint8:
            wav = (wav - 128) / 128.0
    else:
        wav = librosa.load(path, sr=sample_rate)[0]

    wav = wav.astype(np.float32)

    return sr, wav


def pitch(y, hp):
    # Extract Pitch/f0 from raw waveform using PyWORLD
    y = y.astype(np.float64)
    """
    f0_floor : float
        Lower F0 limit in Hz.
        Default: 71.0
    f0_ceil : float
        Upper F0 limit in Hz.
        Default: 800.0
    """
    f0, timeaxis = pw.dio(
        y,
        hp.sample_rate,
        frame_period=hp.hop_length / hp.sample_rate * 1000,
    )  # For hop size 256 frame period is 11.6 ms
    return f0  # (Number of Frames) = (654,)
