

from glob import glob
import os
from pathlib import Path
import jax.numpy as jnp
import jax
from tqdm import tqdm

from music_utils.midi.midi_encoder import MidiEncoder


class PianoAriaDataloader:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.encoder = MidiEncoder()
        
        self.files = glob(os.path.join(self.folder_path, "**/*.mid"), recursive=True)
        self.file_sizes = []
        for f in tqdm(self.files, "Grouping files..."):
            self.file_sizes.append(os.path.getsize(f))

        self.files, self.file_sizes = zip(*sorted(zip(self.files, self.file_sizes), key=lambda x: x[1]))


    def _pad(self, token_ids, length):
        assert len(token_ids) <= length, "Token IDs length exceeds sequence length"
        padded = jnp.zeros(length, dtype=jnp.int32)
        token_ids = jnp.array(token_ids, dtype=jnp.int32)
        padded = padded.at[:len(token_ids)].set(token_ids)
        return padded

    def load_data(self, batch_size=8, batch_shuffle=True, key=jax.random.PRNGKey(0)):
        """
        Load data in batches. 
        Pads to max length in the batch. 
        Shuffles within the batch if specified.
        """
        start_idx = 0
        while start_idx < len(self.files):
            batch_files = self.files[start_idx:start_idx+batch_size]
            
            batch_tokens = []
            for i in range(len(batch_files)):
                batch_tokens.append(self.encoder.encode(midi_file_path=batch_files[i]))

            max_len = max([len(tokens) for tokens in batch_tokens])

            file_names = []

            for i, file in enumerate(batch_files):
                # Pad to max length in this batch
                padded = self._pad(batch_tokens[i], max_len)
                batch_tokens[i] = padded
                file_names.append(Path(file).name)

            if batch_shuffle:
                # Shuffle within the batch
                perm = jnp.array(jax.random.permutation(key, len(batch_tokens)), dtype=jnp.int32)
                batch_tokens = [batch_tokens[i] for i in perm]
                file_names = [file_names[i] for i in perm]

            yield jnp.array(batch_tokens, dtype=jnp.int32), file_names

            start_idx += batch_size
