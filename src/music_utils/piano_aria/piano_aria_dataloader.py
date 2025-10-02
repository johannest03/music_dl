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
    Loads data lazily to avoid memory issues.
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
        
        # Pre-count segments per file without storing data
        self.segment_refs = []  # List of (file_idx, seg_idx) tuples for all segments
        for f_idx, f in tqdm(enumerate(self.files), "Counting segments..."):
            midi_file = Score(f)
            token_sequences = self.tokenizer(midi_file)

            segments_nrs = self.segmenter.count_segments(token_sequences.ids)

            for s_idx in range(segments_nrs):
                self.segment_refs.append((f_idx, s_idx))
        
        self.length = len(self.segment_refs)
        assert self.length >= len(self.files), "Less segments created from files"
    
    def vocab_size(self):
        return len(self.tokenizer.vocab)

    def load_data(self, batch_size=32, shuffle=True, key=jax.random.PRNGKey(0)):
        """
        Load data in batches lazily.
        Pads to max length in the batch.
        Shuffles at segment level if specified.
        """
        indices = jnp.arange(self.length)
        if shuffle:
            indices = jax.random.permutation(key, indices)
        indices = list(indices)
        start_idx = 0
        while start_idx < self.length:
            batch_indices = indices[start_idx:start_idx+batch_size]
            batch_segments = []
            batch_file_names = []
            for idx in batch_indices:
                f_idx, s_idx = self.segment_refs[idx]
                midi_file = Score(self.files[f_idx])
                token_sequences = self.tokenizer(midi_file)
                segments = self.segmenter.segment(token_sequences.ids)
                batch_segments.append(segments[s_idx])
                batch_file_names.append(Path(self.files[f_idx]).name)
            yield jnp.array(batch_segments, dtype=jnp.int32), batch_file_names
            start_idx += batch_size