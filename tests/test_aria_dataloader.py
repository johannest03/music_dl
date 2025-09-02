
from music_utils.piano_aria.piano_aria_dataloader import PianoAriaDataloader
from params import INPUT_PATH

def test_piano_aria_dataloader():
    dataloader = PianoAriaDataloader(folder_path=INPUT_PATH)
    for batch, file_names in dataloader.load_data(batch_size=8):

        assert len(file_names) == 8
        assert batch.shape[0] == 8
        break
