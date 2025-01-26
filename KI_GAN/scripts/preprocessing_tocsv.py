import pandas as pd
from enum import IntEnum
import numpy as np
import os

class AgentType(IntEnum):
    pedestrian = 0
    animal = 0
    car = 1
    truck = 2
    bus = 3
    motorcycle = 4
    tricycle = 5
    bicycle = 6

class TrafficSignalType(IntEnum):
    red = 0
    yellow = 3
    green = 1

class EncodedTrafficSignal(IntEnum):
    GGRR = 0
    YYRR = 1
    RRRR = 2
    RRGG = 3
    RRYY = 4

def encode_traffic_signals(signals: np.ndarray):
    # Convert signals to integers
    signals = list(int(e) for e in signals)
    if (signals == [
            TrafficSignalType.green, 
            TrafficSignalType.red, 
            TrafficSignalType.red, 
            TrafficSignalType.green,
            TrafficSignalType.green,
            TrafficSignalType.red,
            TrafficSignalType.red,
            TrafficSignalType.green
        ]):
        return EncodedTrafficSignal.GGRR
    elif (signals == [
            TrafficSignalType.yellow, 
            TrafficSignalType.red, 
            TrafficSignalType.red, 
            TrafficSignalType.yellow,
            TrafficSignalType.yellow,
            TrafficSignalType.red,
            TrafficSignalType.red,
            TrafficSignalType.yellow
        ]):
        return EncodedTrafficSignal.YYRR
    elif (signals == [
            TrafficSignalType.red, 
            TrafficSignalType.red, 
            TrafficSignalType.red, 
            TrafficSignalType.red,
            TrafficSignalType.red,
            TrafficSignalType.red,
            TrafficSignalType.red,
            TrafficSignalType.red
        ]):
        return EncodedTrafficSignal.RRRR
    elif (signals == [
            TrafficSignalType.red, 
            TrafficSignalType.green, 
            TrafficSignalType.green, 
            TrafficSignalType.red,
            TrafficSignalType.red,
            TrafficSignalType.green,
            TrafficSignalType.green,
            TrafficSignalType.red
        ]):
        return EncodedTrafficSignal.RRGG
    elif (signals == [
            TrafficSignalType.red, 
            TrafficSignalType.yellow, 
            TrafficSignalType.yellow, 
            TrafficSignalType.red,
            TrafficSignalType.red,
            TrafficSignalType.yellow,
            TrafficSignalType.yellow,
            TrafficSignalType.red
        ]):
        return EncodedTrafficSignal.RRYY
    else:
        raise ValueError("Unknown traffic signal pattern")


def find_interval_index(frame_id, raw_frame_ids):
    # Find the interval index for the given frame_id * 3
    return (frame_id * 3 > raw_frame_ids).sum() - 1

def transform_csv(vehicle_file: str, pedestrian_file: str, traffic_file: str, output_file: str):
    # Read the vehicle CSV file
    vehicle_df = pd.read_csv(vehicle_file)
    pedestrian_df = pd.read_csv(pedestrian_file)
    # Remove the "P" and convert the remaining part to an integer using vectorized string operations
    pedestrian_df['track_id'] = pedestrian_df['track_id'].str[1:].astype(int) + vehicle_df['track_id'].max() + 1
    pedestrian_df['length'] = (9.96 + 9.65) / 2 * 2.5 / 100 # chest depth (m)
    pedestrian_df['width']  = (20.04 + 17.72) / 2 * 2.5 / 100 # shoulder breadth (m)

    # Select and reorder the relevant columns
    vehicle_df = vehicle_df[['frame_id', 'track_id', 'x', 'y', 'vx', 'vy', 'ax', 'ay', 'length', 'width', 'agent_type']]
    pedestrian_df = pedestrian_df[['frame_id', 'track_id', 'x', 'y', 'vx', 'vy', 'ax', 'ay', 'length', 'width', 'agent_type']]
    
    vehicle_df = pd.concat([vehicle_df, pedestrian_df], ignore_index=True)
    
    # Map the agent_type to its corresponding integer value
    vehicle_df['agent_type'] = vehicle_df['agent_type'].apply(lambda x: AgentType[x].value)
    
    # Read the traffic light CSV file
    traffic_df = pd.read_csv(traffic_file)
    
    # Transform RawFrameID to frame_id and sort
    traffic_df['frame_id'] = traffic_df['RawFrameID'] // 3
    raw_frame_ids = traffic_df['RawFrameID'].values
    
    # Encode traffic signals
    traffic_df['encoded_traffic_signal'] = traffic_df.apply(
        lambda row: encode_traffic_signals(row[[f'Traffic light {i+1}' for i in range(8)]].values), axis=1)
    
    # Determine the interval for each row in vehicle_df
    traffic_frame_index = vehicle_df['frame_id'].apply(lambda x: find_interval_index(x, raw_frame_ids))
    
    # Map the traffic index to the corresponding encoded traffic signal
    vehicle_df['encoded_traffic_signal'] = traffic_frame_index.apply(lambda idx: traffic_df.iloc[idx]['encoded_traffic_signal'] if idx >= 0 else None)
    
    # Save the transformed DataFrame to a new CSV file
    vehicle_df.to_csv(output_file, index=False, header=False)
    
# Function to process each folder and apply transform_csv
def process_folders(base_path, output_path):
    # Iterate through each subfolder in the base path
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        
        if os.path.isdir(folder_path):
            vehicle_file = os.path.join(folder_path, 'Veh_smoothed_tracks.csv')
            pedestrian_file = os.path.join(folder_path, 'Ped_smoothed_tracks.csv')
            traffic_file_name = next((f for f in os.listdir(folder_path) if f.startswith('Traffic')), None)
            traffic_file = os.path.join(folder_path, f'{traffic_file_name}')
            output_file = os.path.join(output_path, f'Veh_Ped_Traffic_{folder_name}.csv')
            
            # Check if all required files exist
            if os.path.exists(vehicle_file) and os.path.exists(pedestrian_file) and os.path.exists(traffic_file):
                print(f'Processing folder: {folder_name}')
                transform_csv(vehicle_file, pedestrian_file, traffic_file, output_file)
            else:
                print(f'Skipping folder {folder_name} due to missing files')

script_dir = os.path.dirname(__file__)
# Base path where the folders are located
base_path = os.path.join(script_dir, '..', '..', 'SinD', 'data') 

# Output path where the combined CSV files will be saved
output_path = os.path.join(script_dir, '..', 'datasets', 'train_val_no_split') 

# Create the output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)
os.makedirs(os.path.join(script_dir, '..', 'datasets', 'Tianjin', 'train'), exist_ok=True)
os.makedirs(os.path.join(script_dir, '..', 'datasets', 'Tianjin', 'val'), exist_ok=True)
# Process the folders and create combined CSV files
process_folders(base_path, output_path)
