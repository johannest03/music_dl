

import os
from pathlib import Path
import jax.numpy as jnp
import jax
from tqdm import tqdm

from music_utils.midi.midi_segmentation import MidiSegmentation
from params import MAX_SEQUENCE_LENGTH, tokenizer_config, vocab_size
from symusic import Score
from miditok import MMM
class PianoAriaDataloader:
    """
    DataLoader for Piano Aria dataset.
    """
    def __init__(self, files):
        self.files = files
        
        if not Path("tokenizer.json").exists():
            self.tokenizer = MMM(
                tokenizer_config=tokenizer_config
            )
            print("Training tokenizer...")
            midi_paths = [str(Path(f).resolve()) for f in self.files] 
            self.tokenizer.train(vocab_size=vocab_size, files_paths=midi_paths)
            self.tokenizer.save("tokenizer.json")
        else:
            self.tokenizer = MMM(tokenizer_config=tokenizer_config, params="tokenizer.json")

        self.segmenter = MidiSegmentation(max_sequence_length=MAX_SEQUENCE_LENGTH, pad_token_id=self.tokenizer.pad_token_id)
        self.segments = []
        self.file_names = []
        for f in tqdm(self.files, "Segmenting files..."):
            midi_file = Score(f)
            token_sequences = self.tokenizer(midi_file) 
            segments = self.segmenter.segment(token_sequences.ids)
            self.segments.extend(segments)
            self.file_names.extend([Path(f).name] * len(segments))
          
        assert len(self.segments) == len(self.file_names), "Segments and file names length mismatch"
        assert len(self.segments) >= len(self.files), "Less segments created from files"
        self.length = len(self.segments)
    
    def vocab_size(self):
        return len(self.tokenizer.vocab)

    def load_data(self, batch_size=32, shuffle=True, key=jax.random.PRNGKey(0)):
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