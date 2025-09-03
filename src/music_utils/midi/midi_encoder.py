from mido import MidiFile
from params import CONTROL_BINS_CONTROL_DISTANCE, CONTROL_BINS_VALUE_DISTANCE, TIME_BINS_DISTANCE, TIME_RESOLUTION, TEMPO_BINS_DISTANCE
from .midi_token_id_conversion import MidiTokenIDConversion

class MidiEncoder:
    def __init__(self, ticks_per_beat=480):
        self.ticks_per_beat = ticks_per_beat
        self.converter = MidiTokenIDConversion()

    def encode(self, midi_file_path=None, midi_file=None):
        try:
            if midi_file_path is not None:
                midi = MidiFile(midi_file_path)
            elif midi_file is not None:
                midi = midi_file
            else:
                raise ValueError("Either midi_file_path or midi_file must be provided.")

            tokens = ["SONG_START"]

            for track in midi.tracks:
                tokens.append("TRACK_START")
                time_shifts = 0

                for msg in track:
                    # Convert delta ticks -> ms
                    if msg.time > 0:
                        n_steps = int(msg.time // TIME_RESOLUTION)
                        if n_steps > 0:
                            for _ in range(n_steps):
                                tokens.append(f"TIME_SHIFT_{TIME_RESOLUTION}")
                                time_shifts += TIME_RESOLUTION
                    if msg.type == "note_on":
                        tokens.append(f"VELOCITY_{msg.velocity}")
                        tokens.append(f"TIME_{int(round((msg.time - time_shifts) / TIME_BINS_DISTANCE) * TIME_BINS_DISTANCE)}")
                        time_shifts = 0
                        tokens.append(f"NOTE_ON_{msg.note}")
                    elif msg.type == "note_off":
                        tokens.append(f"NOTE_OFF_{msg.note}")
                    elif msg.type == "program_change":
                        tokens.append(f"PROGRAM_{msg.program}")
                    elif msg.type == "control_change":
                        tokens.append(f"TIME_{(msg.time - time_shifts)}")
                        tokens.append(f"CONTROL_{int(round(msg.control / CONTROL_BINS_CONTROL_DISTANCE) * CONTROL_BINS_CONTROL_DISTANCE)}_{int(round(msg.value / CONTROL_BINS_VALUE_DISTANCE) * CONTROL_BINS_VALUE_DISTANCE )}")
                        time_shifts = 0
                    elif msg.type == "set_tempo":
                        tempo = msg.tempo
                        bpm = int(round(60000000 / tempo / TEMPO_BINS_DISTANCE) * TEMPO_BINS_DISTANCE)
                        tokens.append(f"TEMPO_{bpm}")
                    else : 
                        pass

                tokens.append("TRACK_END")
            tokens.append("SONG_END")
            return self.converter.tokens_to_ids(tokens)

        except Exception as e:
            print(f"Error encoding MIDI: {e}")
            return []

    def vocab_size(self):
        return self.converter.vocab_size()