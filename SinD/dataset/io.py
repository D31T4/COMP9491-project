import os
import pandas as pd
import numpy as np
import math

from SinD.dataset.type import AgentType, CrossType, SignalViolationBehavior

def get_dataset_records(path: str) -> list[str]:
    '''
    get dataset record folders in `path`

    Args:
    ---
    - path: dataset path

    Returns:
    ---
    - list of dataset folders
    '''
    return [f for f in os.listdir(path) if os.path.isdir(f'{path}/{f}')]

def read_record_meta(path: str) -> pd.DataFrame:
    '''
    read record metadata

    Args:
    ---
    - path
    '''
    
    dtype={
        'RecordingID': int,
        'City': str,
        'RecordWeekday': str,
        'RecordTimePeriod': str,
        'Weather': str,
        'RawFrameRate': np.float64,
        'RecordDuration': str,
        'Tps': int,
        'car': int,
        'truck': int,
        'bus': int,
        'bicycle': int,
        'motorcycle': int,
        'tricycle': int,
        'pedestrian': int
    }

    return pd.read_csv(path, names=[*dtype.keys()], dtype=dtype, header=0)

def read_pedestrian_trajectories(path: str) -> pd.DataFrame:
    dtype = {
        'track_id': str,
        'frame_id': int,
        'timestamp': np.float64,
        'agent_type': str,
        'x': np.float64,
        'y': np.float64,
        'vx': np.float64,
        'vy': np.float64,
        'ax': np.float64,
        'ay': np.float64
    }

    df = pd.read_csv(path, names=[*dtype.keys()], dtype=dtype, header=0)
    
    del df['agent_type']

    return df

def read_pedestrian_meta(path: str) -> pd.DataFrame:
    dtype = {
        'track_id': str,
        'initialFrame': int,
        'finalFrame': int,
        'Frame_nums': int,
        'agent_type': str
    }

    df = pd.read_csv(path, names=[*dtype.keys()], dtype=dtype, header=0)

    df['agent_type'] = df['agent_type'].apply(lambda val: AgentType[val])

    return df

def read_vehicle_trajectories(path: str) -> pd.DataFrame:
    dtype = {
        'track_id': str,
        'frame_id': int,
        'timestamp': np.float64,
        'agent_type': str,
        'x': np.float64,
        'y': np.float64,
        'vx': np.float64,
        'vy': np.float64,
        'yaw_rad': np.float64,
        'heading_rad': np.float64,
        'length': np.float64,
        'width': np.float64,
        'ax': np.float64,
        'ay': np.float64,
        'v_lon': np.float64,
        'v_lat': np.float64,
        'a_lon': np.float64,
        'a_lat': np.float64
    }

    df = pd.read_csv(path, names=[*dtype.keys()], dtype=dtype, header=0)

    df.drop(columns=['agent_type', 'length', 'width'], inplace=True)

    return df

def read_vehicle_meta(path: str) -> pd.DataFrame:
    dtype = {
        'track_id': str,
        'initialFrame': int,
        'finalFrame': int,
        'Frame_nums': int,
        'width': np.float64,
        'length': np.float64,
        'agent_type': str,
        'CrossType': str,
        'Signal_Violation_Behavior': str
    }

    df = pd.read_csv(path, names=[*dtype.keys()], dtype=dtype, header=0)

    df['agent_type'] = df['agent_type'].apply(lambda val: AgentType[val.strip()])
    df['CrossType'] = df['CrossType'].apply(lambda val: CrossType[val.strip()])
    df['Signal_Violation_Behavior'] = df['Signal_Violation_Behavior'].apply(lambda val: SignalViolationBehavior.parse(val.strip()))

    return df

def read_traffic_signals(path: str) -> pd.DataFrame:
    dtype = {
        'frame_id': int,
        'timestamp': np.float64,
        's1': int,
        's2': int,
        's3': int,
        's4': int,
        's5': int,
        's6': int,
        's7': int,
        's8': int
    }

    df = pd.read_csv(path, names=[*dtype.keys()], dtype=dtype, header=0)

    # frame = raw_frame / 3
    df['frame_id'] = df['frame_id'].apply(lambda x: math.ceil(x / 3)).astype(int)

    return df

def read_map(path: str):
    raise NotImplementedError()