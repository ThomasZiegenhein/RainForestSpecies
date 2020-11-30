from abc import abstractmethod, ABC
import os
import random

from mxnet.gluon.data import dataset
from mxnet import nd
import pandas as pd
from interval import Interval

from src.helpfun.static_help_fun import Transforms, HandleSounds
from src.helpfun.constants import N_FFT, T_SNIPPET_LENGTH, NUM_SPECIES


class DatasetRFS(dataset.Dataset):
    """
    Usage: provide a root directory that has the following structure:
        root/
           | - train_tp.csv
           | - train/
                    | - *.flac

    Parameters
    ----------
        :param root : root folder as String as explained above

    """

    def __init__(self, root):
        super(DatasetRFS)
        self.root = root
        self.snippet_list = []

        true_pos_df = pd.read_csv(os.path.join(root, 'train_tp.csv'))
        true_pos_df['duration'] = true_pos_df.t_max - true_pos_df.t_min
        true_pos_df['bandwidth'] = true_pos_df.f_max - true_pos_df.f_min

        for i in range(len(true_pos_df)):
            self.snippet_list.append(
                FactorySoundSnippets.get_sound_snippet(root=os.path.join(self.root, 'train'), df=true_pos_df, idx_row=i))
            self.snippet_list.append(
                FactorySoundSnippets.get_sound_snippet(root=os.path.join(self.root, 'train'), df=true_pos_df, idx_row=i,
                                                       random_snippet=True, min_overlap=0.33))
            self.snippet_list.append(
                FactorySoundSnippets.get_sound_snippet(root=os.path.join(self.root, 'train'), df=true_pos_df, idx_row=i,
                                                       random_snippet=True, min_overlap=0.33))
            self.snippet_list.append(
                FactorySoundSnippets.get_sound_snippet(root=os.path.join(self.root, 'train'), df=true_pos_df, idx_row=i,
                                                       random_snippet=True, min_overlap=None))
            self.snippet_list.append(
                FactorySoundSnippets.get_sound_snippet(root=os.path.join(self.root, 'train'), df=true_pos_df, idx_row=i,
                                                       random_snippet=True, min_overlap=None))
            self.snippet_list.append(
                FactorySoundSnippets.get_sound_snippet(root=os.path.join(self.root, 'train'), df=true_pos_df, idx_row=i,
                                                       random_snippet=True, min_overlap=None))

    def __len__(self):
        return len(self.snippet_list)

    def __getitem__(self, idx):
        return nd.expand_dims(self.snippet_list[idx].get_input(), axis=0), self.snippet_list[idx].get_result()


class SoundSnippet(ABC):
    """
    Abstract class to extract a sound snippet from the audio input. The 60s sound files contain the song from birds,
    which are between 1 and 10 seconds long. This class extracts a snippet of a certain length from the 60s sound file.

    Parameters
    ----------
        :param file: path to flac sound file
        :param time_song_start: ~
        :param time_song_end: ~
        :param species_id: ~

    """

    def __init__(self, file, time_song_start, time_song_end, species_id):
        self.file = file
        self.song_start = time_song_start
        self.song_end = time_song_end
        self.species_id = species_id
        self.time_start = -1
        self.time_end = -1

    @abstractmethod
    def get_transform(self):
        """Calculate the transformation of the sound wave form"""
        raise NotImplementedError("Not Implemented")

    def get_overlap(self):
        """Calculate the overlap between song snippet from training and this snippet"""
        song = Interval(self.song_start, self.song_end)
        snippet = Interval(self.time_start, self.time_end)
        overlap = song & snippet
        length_song = song.upper_bound - song.lower_bound
        length_snippet = snippet.upper_bound - snippet.lower_bound
        fraction = min(1, (overlap.upper_bound - overlap.lower_bound) / min(length_song, length_snippet))
        return fraction

    def get_input(self):
        """Data that is given to the data handler as input for the training"""
        return nd.array(self.get_transform())

    def get_result(self):
        """Data that is given to the data handler as result for the training"""
        vector_output = nd.zeros(NUM_SPECIES+1)
        if self.get_overlap() > 0.5:
            vector_output[self.species_id] = 1
        else:
            vector_output[-1] = 1
        return vector_output


class SnippetBirds(SoundSnippet):
    """Puts bird song in the center of the extracted sound snippet"""

    def get_transform(self):
        self.time_start = max((self.song_start + self.song_end) / 2 - T_SNIPPET_LENGTH / 2, 0.0)
        self.time_end = self.time_start + T_SNIPPET_LENGTH
        x, sr = HandleSounds.load_sound(self.file, start=self.time_start, end=self.time_end)
        return Transforms.get_sum_FFT(x, N_FFT)


class SnippetRandom(SoundSnippet):
    """Randomly extract sound snippet"""

    def get_transform(self):
        self.time_start = random.uniform(0, 1) * (60.0 - T_SNIPPET_LENGTH)
        self.time_end = self.time_start + T_SNIPPET_LENGTH
        x, sr = HandleSounds.load_sound(self.file, start=self.time_start, end=self.time_end)
        return Transforms.get_sum_FFT(x, N_FFT)


class SnippetRandomOverlap(SoundSnippet):
    """Extract sound snippet with overlap"""

    def __init__(self, file, time_song_start, time_song_end, song_species_id, min_overlap):
        super(SnippetRandomOverlap, self).__init__(file, time_song_start, time_song_end, song_species_id)
        self.min_overlap = min_overlap

    def get_transform(self):
        time_overlap = max(random.uniform(0, 1), self.min_overlap)
        if random.uniform(-0.5, 0.5) > 0:
            self.time_start = max(self.song_start - (T_SNIPPET_LENGTH * (1 - time_overlap)), 0)
        else:
            self.time_start = max(self.song_end - (T_SNIPPET_LENGTH * time_overlap), 0)
        self.time_end = self.time_start + T_SNIPPET_LENGTH
        x, sr = HandleSounds.load_sound(self.file, start=self.time_start, end=self.time_end)
        return Transforms.get_sum_FFT(x, N_FFT)


class FactorySoundSnippets:
    """ Factory for SoundSnippet class with certain context, depending on the input parameters"""
    @staticmethod
    def get_sound_snippet(root, df, idx_row, random_snippet=False, min_overlap=None) -> SoundSnippet:
        """
        Factory body

        Parameters
        ----------
        :param root: Path to folder with sound files
        :param df: Data Format from pandas with the training information
        :param idx_row: Row of interest
        :param random_snippet: Extract random snipper from sound input?
        :param min_overlap: Minimal overlap of the bird song and extracted snipped
        """
        file = os.path.join(root, df.iloc[idx_row].recording_id + ".flac")
        time_song_start = df.iloc[idx_row].t_min
        time_song_end = df.iloc[idx_row].t_max
        song_species_id = df.iloc[idx_row].species_id
        if not random_snippet and min_overlap is None:
            return SnippetBirds(file, time_song_start, time_song_end, song_species_id)
        elif random_snippet and min_overlap is None:
            return SnippetRandom(file, time_song_start, time_song_end, song_species_id)
        else:
            return SnippetRandomOverlap(file, time_song_start, time_song_end, song_species_id, min_overlap)