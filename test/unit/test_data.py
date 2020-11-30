import pathlib
import os

from src.helpfun.constants import NUM_SPECIES


def test_data_handler():
    from src.data.data_handler import DatasetRFS
    path_sample_data = os.path.join(pathlib.Path(__file__).parent.parent.parent.absolute(), 'sampledata')
    data = DatasetRFS(path_sample_data)
    power_spectra, target = data.__getitem__(0)
    assert power_spectra.shape == (1, 8 * 1024 + 1) and len(target) == NUM_SPECIES+1


test_data_handler()