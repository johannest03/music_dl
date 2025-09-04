
from music_utils.midi.midi_token_id_conversion import MidiTokenIDConversion


class MidiSegmentation:
    def __init__(self, max_sequence_length):
        self.max_sequence_length = max_sequence_length
        self.converter = MidiTokenIDConversion()

    def segment(self, token_ids):
        tokens = [token_ids[i:i + self.max_sequence_length-1] for i in range(0, len(token_ids), self.max_sequence_length-1)]
        segments = []
        for chunk in tokens:
            chunk = chunk + [self.converter.end_token_id()]
            # Pad if needed
            if len(chunk) < self.max_sequence_length:
                pad_token = self.converter.pad_token_id()
                chunk += [pad_token] * (self.max_sequence_length - len(chunk))
            segments.append(chunk)
        return segments