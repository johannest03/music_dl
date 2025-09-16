from params import CONTROL_MAX_CONTROL, CONTROL_MAX_VALUE, TIME_RESOLUTION, TEMPO_BINS_DISTANCE, TIME_BINS_DISTANCE, MAX_TEMPO, CONTROL_BINS_VALUE_DISTANCE, CONTROL_BINS_CONTROL_DISTANCE

class MidiTokenIDConversion:
    def __init__(self):
         # Build simple token->id mappings
        self.token_to_id = {}
        self.id_to_token = []

        # Special tokens
        for t in ["SONG_END", "SONG_START", "TRACK_START", "TRACK_END"]:
            self._add_token(t)

        # Note events
        for note in range(128):
            self._add_token(f"NOTE_ON_{note}")
            self._add_token(f"NOTE_OFF_{note}")

        # Velocity bins
        for v in range(128):
            self._add_token(f"VELOCITY_{v}")
        
        for t in range(0, TIME_RESOLUTION + 1, TIME_BINS_DISTANCE):
            self._add_token(f"TIME_{t}")

        # Program changes
        for p in range(128):
            self._add_token(f"PROGRAM_{p}")

        # Control changes
        for c in range(0, CONTROL_MAX_CONTROL, CONTROL_BINS_CONTROL_DISTANCE):
            for v in range(0, CONTROL_MAX_VALUE + 1, CONTROL_BINS_VALUE_DISTANCE):
                self._add_token(f"CONTROL_{c}_{v}")

        # Tempo bins
        for b in range(0, MAX_TEMPO + 1, TEMPO_BINS_DISTANCE):
            self._add_token(f"TEMPO_{b}")

        # Time shift
        self._add_token(f"TIME_SHIFT_{TIME_RESOLUTION}")
        
        # Padding
        self._add_token(f"PAD")

    def _add_token(self, token):
        self.token_to_id[token] = len(self.id_to_token)
        self.id_to_token.append(token)
        
    def tokens_to_ids(self, tokens):
        ids = [int(self.token_to_id[t]) for t in tokens if t in self.token_to_id]

        for t in tokens:
            assert t in self.token_to_id, f"Token '{t}' not recognized."

        assert len(ids) == len(tokens), "Some tokens were not recognized."
        return ids

    def ids_to_tokens(self, ids):
        return [self.id_to_token[i] for i in ids]
    
    def vocab_size(self):
        return len(self.id_to_token)

    def end_token_id(self):
        return self.token_to_id["SONG_END"]

    def pad_token_id(self):
        return self.token_to_id["PAD"]
    
    def start_token_id(self):
        return self.token_to_id["SONG_START"]
    
    def track_start_token_id(self):
        return self.token_to_id["TRACK_START"]
    
    
    def get_note_ids(self):
        return [self.token_to_id[f"NOTE_ON_{n}"] for n in range(128)] + [self.token_to_id[f"NOTE_OFF_{n}"] for n in range(128)]