from enum import Enum, auto

class FileMode(Enum):
    UNSELECTED = auto()
    APPEND = auto()
    OVERWRITE = auto()

class DataMultiplicity(Enum):
    TRACKERS = auto() # multiple trackers, 1 video
    VIDEOS = auto() # multiple videos, 1 tracker each
    SINGLE = auto() # 1 tracker, 1 video
    BOTH = auto() # multiple videos each with multiple trackers

class TimeUnits(Enum):
    UNSELECTED = auto()
    SECONDS = 's'
    MINUTES = 'min'
    HOURS = 'hr'

class TrackingOperation(Enum):
    UNSELECTED = auto()
    MARKERS = auto()
    NECKING = auto()
    AREA = auto()

class TrackerChoice(Enum):
    UNSELECTED = auto()
    KCF = auto()
    CSRT = auto()