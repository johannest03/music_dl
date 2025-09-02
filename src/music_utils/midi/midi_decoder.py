from mido import MidiFile, MidiTrack, Message, MetaMessage, bpm2tempo
from .midi_token_id_conversion import MidiTokenIDConversion
from params import TEMPO_BINS_DISTANCE  

class MidiDecoder:
    def __init__(self, ticks_per_beat=480):
        self.ticks_per_beat = ticks_per_beat
        self.converter = MidiTokenIDConversion()

    def decode(self, tokens, output_path=None):
        tokens = self.converter.ids_to_tokens(tokens)
        midi = MidiFile(ticks_per_beat=self.ticks_per_beat)
        track = None
        tempo = 500000
        delta_ticks = 0
        current_velocity = 64  # default mid velocity

        for token in tokens:
            if token == "SONG_START":
                continue
            elif token == "SONG_END":
                break
            elif token == "TRACK_START":
                track = MidiTrack()
                midi.tracks.append(track)
                delta_ticks = 0
            elif token == "TRACK_END":
                track = None
            elif token.startswith("NOTE_ON_"):
                note = int(token.split("_")[2])
                track.append(Message("note_on", note=note, velocity=current_velocity, time=delta_ticks))
                delta_ticks = 0
            elif token.startswith("NOTE_OFF_"):
                note = int(token.split("_")[2])
                track.append(Message("note_off", note=note, velocity=0, time=delta_ticks))
                delta_ticks = 0
            elif token.startswith("VELOCITY_"):
                current_velocity = int(token.split("_")[1])
            elif token.startswith("PROGRAM_"):
                program = int(token.split("_")[1])
                track.append(Message("program_change", program=program, time=delta_ticks))
                delta_ticks = 0
            elif token.startswith("TEMPO_"):
                bpm = int(token.split("_")[1])
                tempo = bpm2tempo(bpm * TEMPO_BINS_DISTANCE) 
                track.append(MetaMessage("set_tempo", tempo=tempo, time=delta_ticks))
                delta_ticks = 0
            elif token.startswith("TIME_SHIFT_"):
                delta_ticks += int(token.split("_")[-1])
            elif token.startswith("TIME_"):
                time = int(token.split("_")[1])
                delta_ticks += time
            elif token.startswith("CONTROL_"):
                control, value = map(int, token.split("_")[1:])
                track.append(Message("control_change", control=control, value=value, time=delta_ticks))
                delta_ticks = 0
            else:
                print(f"Unknown token: {token}")

        if output_path:
            midi.save(output_path)
            
        return midi

