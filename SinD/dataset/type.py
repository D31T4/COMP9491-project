from enum import IntEnum

class AgentType(IntEnum):
    pedestrian = 0
    # there is 1 instance of animal in dataset...
    # just categorize it as pedestrian...
    animal = 0
    car = 1
    truck = 2
    bus = 3
    motorcycle = 4
    tricycle = 5
    bicycle = 6

class CrossType(IntEnum):
    StraightCross = 0
    LeftTurn = 1
    RightTurn = 2
    Others = 3
    NoRecord = -1

class TrafficSignalType(IntEnum):
    red = 0
    yellow = 3
    green = 1

class EncodedTrafficSignal(IntEnum):
    '''
    encoded traffic signals
    see: table 1 in https://arxiv.org/abs/2404.11181
    '''
    GGRR = 0
    YYRR = 1
    RRRR = 2
    RRGG = 3
    RRYY = 4

class SignalViolationBehavior(IntEnum):
    red_light_running = 0
    yellow_light_running = 1
    no_violation = 2
    no_record = -1

    @staticmethod
    def parse(text: str):
        match text:
            case 'No violation of traffic lights':
                return SignalViolationBehavior.no_violation
            case 'yellow-light running':
                return SignalViolationBehavior.yellow_light_running
            case 'red-light running':
                return SignalViolationBehavior.red_light_running
            case _:
                raise ValueError(f'unexpected value of `text`: {text}')
        
    @staticmethod
    def to_str(enum: int):
        match enum:
            case 0:
                return 'red-light running',
            case 1:
                return 'yellow-light running',
            case 2:
                return 'No violation of traffic lights'
            case _:
                raise ValueError(f'unexpected value of `enum`: {enum}')
