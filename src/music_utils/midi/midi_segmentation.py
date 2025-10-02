class MidiSegmentation:
    def __init__(self, max_sequence_length, pad_token_id):
        self.max_sequence_length = max_sequence_length
        self.pad_token_id = pad_token_id

    def segment(self, token_ids):
        # Restore sliding window with overlap (step by max_sequence_length - 1)
        tokens = [token_ids[i:i + self.max_sequence_length] for i in range(0, len(token_ids), self.max_sequence_length - 1)]
        segments = []
        for chunk in tokens:
            # Pad if shorter than max_sequence_length
            if len(chunk) < self.max_sequence_length:
                chunk += [self.pad_token_id] * (self.max_sequence_length - len(chunk))
            segments.append(chunk)
        return segments
    
    def count_segments(self, token_ids):
        return (len(token_ids) + (self.max_sequence_length - 2)) // (self.max_sequence_length - 1)