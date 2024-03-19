from enum import Enum, auto

class FileMode(Enum):
    APPEND = auto()
    OVERWRITE = auto()

class DataMultiplicity(Enum):
    TRACKERS = auto() # multiple trackers, 1 video
    VIDEOS = auto() # multiple videos, 1 tracker each
    SINGLE = auto() # 1 tracker, 1 video
    BOTH = auto() # multiple videos each with multiple trackers

class TimeUnits(Enum):
    SECONDS = auto()
    MINUTES = auto()
    HOURS = auto()

class TrackingOperation(Enum):
    MARKERS = auto()
    NECKING = auto()
    AREA = auto()

class TrackerChoice(Enum):
    KCF = auto()
    CSRT = auto()

