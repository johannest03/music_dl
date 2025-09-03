
from music_utils.piano_aria.piano_aria_dataloader import PianoAriaDataloader
from params import INPUT_PATH
from music_utils.data_split_utils import split_files


def test_piano_aria_dataloader():

    train_files, test_files = split_files(folder_path=INPUT_PATH, train_size=0.8, test_size=0.2)
    dataloader = PianoAriaDataloader(files=train_files)
    for batch, file_names in dataloader.load_data(batch_size=8):

        assert len(file_names) == 8
        assert batch.shape[0] == 8
        break
