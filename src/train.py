
import jax
import jax.numpy as jnp
from music_utils.midi.midi_decoder import MidiDecoder
from pathlib import Path
from music_utils.piano_aria.piano_aria_dataloader import PianoAriaDataloader
from params import INPUT_PATH, OUTPUT_PATH
from model.trainer import Trainer
from music_utils.data_split_utils import split_files

def __main__():
    
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")

    train_files, test_files = split_files(folder_path=INPUT_PATH, train_size=0.8, test_size=0.2)

    train_dataloader = PianoAriaDataloader(files=train_files)
    test_dataloader = PianoAriaDataloader(files=test_files)

    trainer = Trainer(
        model=None,
        dataloader=train_dataloader,
        validation_dataloader=test_dataloader,
        optimizer=None,
        loss_fn=None,
        log_dir=OUTPUT_PATH / "logs",
        ckpt_dir=OUTPUT_PATH / "checkpoints"
    )

    
    
__main__()
