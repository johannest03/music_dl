

from music_utils.midi.midi_encoder import MidiEncoder
from music_utils.midi.midi_segmentation import MidiSegmentation
from music_utils.midi.midi_token_id_conversion import MidiTokenIDConversion
from params import MAX_SEQUENCE_LENGTH


def test_segmentation_ends():
    file = "/datasets/piano_aria/data/aa/000002_0.mid"
    token_id_converter = MidiTokenIDConversion()
    segmenter = MidiSegmentation(max_sequence_length=MAX_SEQUENCE_LENGTH)
    token_ids = MidiEncoder().encode(midi_file_path=file)
    segments = segmenter.segment(token_ids)
    for segment in segments:
        assert segment[-1] == token_id_converter.end_token_id() or segment[-1] == token_id_converter.pad_token_id()
        assert len(segment) == MAX_SEQUENCE_LENGTH