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
    NECKING_MIDPT = auto()
    AREA = auto()

class TrackerChoice(Enum):
    UNSELECTED = auto()
    KCF = auto()
    CSRT = auto()

class AnalysisType(Enum):
    MARKER_DELTAS = auto()
    NECKING_POINT = auto()
    POISSONS_RATIO = auto()
    MARKER_VELOCITY = auto()
    MARKER_DISPLACEMENT = auto()
    MARKER_DISTANCE = auto()
    SURFACE_AREA = auto()

class LocatorType(Enum):
    BBOX = auto()
    CENTROID = auto()