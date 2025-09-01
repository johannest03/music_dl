from mido import MidiFile
from params import TIME_RESOLUTION

class MidiDecoder():
    
    def __init__(self):
        pass

    def decodes(self, tokens, output_path):
        midi = MidiFile()  # up to 16 channels
        time = 0
        track = 0
        channel = 0

        for token in tokens:
            if token.startswith("TIME_SHIFT"):
                shift = int(token.split("_")[2])
                time += shift / TIME_RESOLUTION  # back to beats

            elif token.startswith("NOTE_ON"):
                parts = token.split("_")
                pitch = int(parts[1][1:])  # after "p"
                vel = int(parts[2][1:])
                midi.addNote(track, channel, pitch, time, 1, vel)

            elif token.startswith("PROGRAM_CHANGE"):
                prog = int(token.split("inst")[1])
                midi.addProgramChange(track, channel, time, prog)

            elif token.startswith("SET_TEMPO"):
                bpm = int(token.split("_")[2])
                midi.addTempo(track, time, bpm)

        with open(output_path, "wb") as f:
            midi.writeFile(f)
