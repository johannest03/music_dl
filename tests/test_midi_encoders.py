import pytest
from music_utils.midi.midi_encoder import MidiEncoder
from music_utils.midi.midi_decoder import MidiDecoder
from mido import MidiFile


def test_midi_encoding_decoding(): # ensures no quality loss due to encoding/decoding on selected songs
    midi_encoder = MidiEncoder()
    midi_decoder = MidiDecoder()

    midi_file_paths = [
        "/datasets/piano_aria/data/aa/000002_0.mid",
        "/datasets/piano_aria/data/aa/000003_0.mid",
        "/datasets/piano_aria/data/aa/000004_0.mid"
    ]
    
    for midi_file_path in midi_file_paths:
        midi = MidiFile(midi_file_path)
        midi_tokens = midi_encoder.encode(midi_file_path=midi_file_path)

        midi_decoded = midi_decoder.decode(tokens = midi_tokens)

        for og_track, track in zip(midi.tracks, midi_decoded.tracks):
            for i, (og_msg, msg) in enumerate(zip(og_track, track)):
                assert og_msg == msg, f"Track {i}: {og_msg} != {msg}"
