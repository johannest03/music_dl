
import jax
import jax.numpy as jnp
from music_utils.midi.midi_encoder import MidiEncoder
from music_utils.midi.midi_decoder import MidiDecoder
from pathlib import Path
from params import OUTPUT_PATH

def __main__():
    
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    
    midi_encoder = MidiEncoder()
    midi_decoder = MidiDecoder()

    midi_file_path = "/datasets/piano_aria/data/aa/000002_0.mid"
    midi_tokens = midi_encoder.encode(midi_file_path=midi_file_path)

    midi_file_out_path = Path(OUTPUT_PATH) / "000002_0.mid"
    midi_decoded = midi_decoder.decode(tokens = midi_tokens, output_path=midi_file_out_path)
    
__main__()
