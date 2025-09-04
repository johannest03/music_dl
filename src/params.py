
INPUT_PATH = "/datasets/piano_aria/data/aa"
OUTPUT_PATH = "/output"

# Music Parameters / Bins
MAX_SEQUENCE_LENGTH = 2000 # will use this length to segment the input data

TIME_RESOLUTION = 100
TIME_BINS_DISTANCE = 10 # used for timing binning

MAX_TEMPO = 300 # in bpm
TEMPO_BINS_DISTANCE = 4 # used for tempo binning

CONTROL_MAX_CONTROL = 64
CONTROL_BINS_CONTROL_DISTANCE = 8 # used for control change binning

CONTROL_MAX_VALUE = 128
CONTROL_BINS_VALUE_DISTANCE = 8 # used for control change binning