
import jax
import jax.numpy as jnp
from music_utils.midi.midi_encoder import MidiEncoder
from music_utils.midi.midi_decoder import MidiDecoder
from pathlib import Path
from music_utils.piano_aria.piano_aria_dataloader import PianoAriaDataloader
from params import INPUT_PATH, OUTPUT_PATH

def __main__():
    
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")


    dataloader = PianoAriaDataloader(folder_path=INPUT_PATH)
    midi_decoder = MidiDecoder()


    for batch, file_names in dataloader.load_data(batch_size=8):
        #print(f"Batch shape: {batch.shape}")

        #midi_file_name = file_names[0]

        #midi_file_out_path = Path(OUTPUT_PATH) / midi_file_name
        
        #print(midi_file_name, batch[0])
        
        #midi_decoded = midi_decoder.decode(tokens = batch[0], output_path=midi_file_out_path)
        pass
    
__main__()
