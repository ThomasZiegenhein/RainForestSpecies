import librosa as lr
import numpy as np


class Transforms:
    @staticmethod
    def get_sum_FFT(float_array, n_fft):
        """
        Windowed (hanning) fft transformation of length n_fft over the float array input.
        Single ffts are summed and normed at in order to get one sepctrum.

        Parameters
        ----------
        :param float_array: Path to folder with sound files
        :param n_fft: length of fft window
        """
        xs = lr.stft(float_array, n_fft=n_fft)
        xdb = (abs(xs))
        db = (np.sum(xdb, axis=1))
        db_min = abs(np.amin(db))
        db_max = abs(np.amax(db))
        norm = max(db_min, db_max)
        db /= norm
        return db


class HandleSounds:
    @staticmethod
    def load_sound(file, start=0.0, end=60.0):
        """
        load sounds as float array from start to end

        Parameters
        ----------
        :param file: path to *.flac file
        :param start: ~
        :param end: ~ (max 60 seconds here)
        """
        x, sr = lr.load(file)
        start_index = int(start * sr)
        end_index = int(end * sr)
        return x[start_index:end_index], sr
