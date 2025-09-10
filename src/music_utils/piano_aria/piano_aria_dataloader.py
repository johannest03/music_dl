

import os
from pathlib import Path
import jax.numpy as jnp
import jax
from tqdm import tqdm

from music_utils.midi.midi_encoder import MidiEncoder
from music_utils.midi.midi_segmentation import MidiSegmentation
from params import MAX_SEQUENCE_LENGTH


class PianoAriaDataloader:
    """
    DataLoader for Piano Aria dataset.
    """
    def __init__(self, files):
        self.files = files
        self.encoder = MidiEncoder()
        self.segmenter = MidiSegmentation(max_sequence_length=MAX_SEQUENCE_LENGTH)

        self.segments = []
        self.file_names = []
        for f in tqdm(self.files, "Segmenting files..."):
            token_ids = self.encoder.encode(midi_file_path=f)
            segments = self.segmenter.segment(token_ids)
            self.segments.extend(segments)
            self.file_names.extend([Path(f).name] * len(segments))
        self.length = len(self.segments)

    def vocab_size(self):
        return self.encoder.vocab_size()

    def load_data(self, batch_size=8, shuffle=True, key=jax.random.PRNGKey(0)):
        """
        Load data in batches. 
        Pads to max length in the batch. 
        Shuffles within the batch if specified.
        """
        indices = jnp.arange(len(self.segments))
        if shuffle:
            indices = jax.random.permutation(key, indices)
        indices = list(indices)
        start_idx = 0
        while start_idx < len(self.segments):
            batch_indices = indices[start_idx:start_idx+batch_size]
            batch_tokens = [self.segments[i] for i in batch_indices]
            batch_file_names = [self.file_names[i] for i in batch_indices]
            yield jnp.array(batch_tokens, dtype=jnp.int32), batch_file_names
            start_idx += batch_size