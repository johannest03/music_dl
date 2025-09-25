from miditok import TokenizerConfig

INPUT_PATH = "/datasets/piano_aria/data/a_medium/aa" # Specifiy your input data path here, works recursively to all subfolders
OUTPUT_PATH = "/output" # folder to save checkpoints, samples, logs, etc.

# Music params
MAX_SEQUENCE_LENGTH = 512

# Tokenizer Config
vocab_size = 600
tokenizer_config = TokenizerConfig(
    pitch_range=(21, 109),
    beat_res={(0, 4): 8, (4, 12): 4},
    num_velocities=32,
    special_tokens=["PAD", "BOS", "EOS", "MASK"],
    use_chords=True,
    use_rests=True,
    use_tempos=True,
    use_time_signatures=True,
    use_programs=True,
    num_tempos=32,
    tempo_range=(40, 250),
    vocab_size=vocab_size,
    base_tokenizer='REMI'
)
