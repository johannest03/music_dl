from mido import MidiFile
from params import TIME_RESOLUTION

class MidiEncoder():
    
    def __init__(self):
        pass

    def encode(self, mid_file_path):
        midi = MidiFile(mid_file_path)
        tokens = ["SONG_START"]

        for i, track in enumerate(midi.tracks):
            tokens.append("TRACK_START")

            current_time = 0
            for msg in track:
                current_time += msg.time

                # Quantize time shifts
                while current_time >= TIME_RESOLUTION:
                    tokens.append(f"TIME_SHIFT_{TIME_RESOLUTION}")
                    current_time -= TIME_RESOLUTION

                if msg.type == "note_on" and msg.velocity > 0:
                    vel = round(msg.velocity / 32) * 32  # bucketed
                    tokens.append(f"NOTE_ON_p{msg.note}_v{vel}")
                elif msg.type in ["note_off", "note_on"] and msg.velocity == 0:
                    tokens.append(f"NOTE_OFF_p{msg.note}")
                elif msg.type == "program_change":
                    tokens.append(f"PROGRAM_CHANGE_inst{msg.program}")
                elif msg.type == "set_tempo":
                    bpm = int(60000000 / msg.tempo)
                    tokens.append(f"SET_TEMPO_{bpm}")

            tokens.append("TRACK_END")

        tokens.append("SONG_END")
        return tokens
