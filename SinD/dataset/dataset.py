import tqdm
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

import SinD.dataset.io as io
from SinD.dataset.type import CrossType, SignalViolationBehavior, AgentType, TrafficSignalType, EncodedTrafficSignal

# https://www.ergocenter.ncsu.edu/wp-content/uploads/sites/18/2017/09/Anthropometric-Summary-Data-Tables.pdf
pedestrian_width = (20.04 + 17.72) / 2 * 2.5 / 100 # shoulder breadth (m)
pedestrian_length = (9.96 + 9.65) / 2 * 2.5 / 100 # chest depth (m)

# feature list
TRAJECTORY_FEATURES = ['x', 'y', 'vx', 'vy', 'ax', 'ay', 'yaw_rad', 'heading_rad', 'v_lon', 'v_lat', 'a_lon', 'a_lat']
AGENT_FEATURES = ['length', 'width']
AGENT_FLAGS = ['agent_type', 'CrossType', 'Signal_Violation_Behavior']
SIGNAL_FEATURES = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']

# feature dimensions
TRAJECTORY_FEATURE_DIM = len(TRAJECTORY_FEATURES)
AGENT_FEATURE_DIM = len(AGENT_FEATURES)
AGENT_FLAG_DIM = len(AGENT_FLAGS)
SIGNAL_DIM = len(SIGNAL_FEATURES)

def encode_traffic_signals(signals: torch.IntTensor) -> torch.IntTensor:
    '''
    encode traffic signals as KI-GAN format
    see table 1 in https://arxiv.org/abs/2404.11181

    Args:
    ---
    - signals: [..., 8]

    Returns:
    ---
    - encoded signals: [..., 1]
    '''
    encoded = torch.zeros((*signals.shape[:-1], 1), dtype=torch.int8).fill_(-1)

    encoded[
        (signals == torch.tensor([
            TrafficSignalType.green, 
            TrafficSignalType.red, 
            TrafficSignalType.red, 
            TrafficSignalType.green,
            TrafficSignalType.green,
            TrafficSignalType.red,
            TrafficSignalType.red,
            TrafficSignalType.green
        ], dtype=torch.int8)).all(dim=-1)
    , :] = EncodedTrafficSignal.GGRR

    encoded[
        (signals == torch.tensor([
            TrafficSignalType.yellow, 
            TrafficSignalType.red, 
            TrafficSignalType.red, 
            TrafficSignalType.yellow,
            TrafficSignalType.yellow,
            TrafficSignalType.red,
            TrafficSignalType.red,
            TrafficSignalType.yellow
        ], dtype=torch.int8)).all(dim=-1)
    , :] = EncodedTrafficSignal.YYRR

    encoded[
        (signals == torch.tensor([
            TrafficSignalType.red, 
            TrafficSignalType.red, 
            TrafficSignalType.red, 
            TrafficSignalType.red,
            TrafficSignalType.red,
            TrafficSignalType.red,
            TrafficSignalType.red,
            TrafficSignalType.red
        ], dtype=torch.int8)).all(dim=-1)
    , :] = EncodedTrafficSignal.RRRR

    encoded[
        (signals == torch.tensor([
            TrafficSignalType.red, 
            TrafficSignalType.green, 
            TrafficSignalType.green, 
            TrafficSignalType.red,
            TrafficSignalType.red,
            TrafficSignalType.green,
            TrafficSignalType.green,
            TrafficSignalType.red
        ], dtype=torch.int8)).all(dim=-1)
    , :] = EncodedTrafficSignal.RRGG

    encoded[
        (signals == torch.tensor([
            TrafficSignalType.red, 
            TrafficSignalType.yellow, 
            TrafficSignalType.yellow, 
            TrafficSignalType.red,
            TrafficSignalType.red,
            TrafficSignalType.yellow,
            TrafficSignalType.yellow,
            TrafficSignalType.red
        ], dtype=torch.int8)).all(dim=-1)
    , :] = EncodedTrafficSignal.RRYY

    assert torch.all(encoded != -1)

    return encoded

def decode_traffic_signals(encoded: torch.IntTensor) -> torch.IntTensor:
    '''
    decod traffic signal from KI-GAN format

    Args:
    ---
    - encoded: encoded signals [..., 1]

    Returns:
    ---
    - decoded signals [..., 8]
    '''
    decoded = torch.zeros((*encoded.shape[:-1], SIGNAL_DIM), dtype=torch.int8)
    
    decoded[encoded[..., 0] == EncodedTrafficSignal.GGRR] = torch.tensor([
        TrafficSignalType.green, 
        TrafficSignalType.red, 
        TrafficSignalType.red, 
        TrafficSignalType.green,
        TrafficSignalType.green,
        TrafficSignalType.red,
        TrafficSignalType.red,
        TrafficSignalType.green
    ], dtype=torch.int8)

    decoded[encoded[..., 0] == EncodedTrafficSignal.YYRR] = torch.tensor([
        TrafficSignalType.yellow, 
        TrafficSignalType.red, 
        TrafficSignalType.red, 
        TrafficSignalType.yellow,
        TrafficSignalType.yellow,
        TrafficSignalType.red,
        TrafficSignalType.red,
        TrafficSignalType.yellow
    ], dtype=torch.int8)

    decoded[encoded[..., 0] == EncodedTrafficSignal.RRRR] = torch.tensor([
        TrafficSignalType.red, 
        TrafficSignalType.red, 
        TrafficSignalType.red, 
        TrafficSignalType.red,
        TrafficSignalType.red,
        TrafficSignalType.red,
        TrafficSignalType.red,
        TrafficSignalType.red
    ], dtype=torch.int8)

    decoded[encoded[..., 0] == EncodedTrafficSignal.RRGG] = torch.tensor([
        TrafficSignalType.red, 
        TrafficSignalType.green, 
        TrafficSignalType.green, 
        TrafficSignalType.red,
        TrafficSignalType.red,
        TrafficSignalType.green,
        TrafficSignalType.green,
        TrafficSignalType.red
    ], dtype=torch.int8)

    decoded[encoded[..., 0] == EncodedTrafficSignal.RRYY] = torch.tensor([
        TrafficSignalType.red, 
        TrafficSignalType.yellow, 
        TrafficSignalType.yellow, 
        TrafficSignalType.red,
        TrafficSignalType.red,
        TrafficSignalType.yellow,
        TrafficSignalType.yellow,
        TrafficSignalType.red
    ], dtype=torch.int8)

    return decoded

def get_traffic_signal_record_name(fname: str):
    '''
    get traffic signal file name

    Args:
    ---
    - fname: file name
    '''
    month, day, seq = fname.split('_')
    return f'{month}_{day.rjust(2, '0')}_{seq}'

class SignalizedIntersectionDatasetConfig:
    '''
    SinD dataset config
    '''
    
    def __init__(
        self, 
        obs_len: int, 
        pred_len: int,
        stride: int = 1,
        min_predictions: int = 1,
        encode_traffic_signals: bool = False,
        padding_value: float = 0,
        include_incomplete_trajectories: bool = True
    ):
        '''
        Args:
        ---
        - obs_len: no. of history frames
        - pred_len: no. of future frames
        - stride: no. of frames to be skipped
        - min_predictions: min. no. of agents in future frames
        - encode_traffic_signals: encode traffic signals into KI-GAN format
        '''
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.stride = stride
        self.min_predictions = min_predictions
        self.encode_traffic_signals = encode_traffic_signals
        self.padding_value = padding_value

        self.include_incomplete_trajectories = include_incomplete_trajectories

    @property
    def seq_len(self) -> int:
        '''
        total sequence length
        '''
        return self.obs_len + self.pred_len
    
    def serialize(self) -> dict[str, any]:
        '''
        serialize to json dict
        '''
        return {
            'obs_len': self.obs_len,
            'pred_len': self.pred_len,
            'stride': self.stride,
            'min_predictions': self.min_predictions
        }
    
    @staticmethod
    def deserialize(json: dict[str, any]):
        '''
        deserialize from json

        Args:
        ---
        - json dict

        Returns:
        ---
        - config instance
        '''
        return SignalizedIntersectionDatasetConfig(
            obs_len=json['obs_len'],
            pred_len=json['pred_len'],
            stride=json['stride'],
            min_predictions=json['min_predictions']
        )


class SignalizedIntersectionDataset(Dataset):
    '''
    SinD trajectory dataset
    '''
    def __init__(self, config: SignalizedIntersectionDatasetConfig):
        '''
        Args:
        ---
        - config
        '''
        super().__init__()

        self.config = config

        # trajectory features
        self.trajectory_records = torch.zeros((0, self.config.seq_len, TRAJECTORY_FEATURE_DIM), dtype=torch.float32)
        self.trajectory_indices: list[tuple[int, int]] = []

        # agent float features
        self.agent_records = torch.zeros((0, AGENT_FEATURE_DIM), dtype=torch.float32)
        # agent categorical features
        self.agent_flags = torch.zeros((0, AGENT_FLAG_DIM), dtype=torch.int8)

        # signal features
        if not self.config.encode_traffic_signals:
            self.signal_records = torch.zeros((0, SIGNAL_DIM), dtype=torch.int8)
        else:
            self.signal_records = torch.zeros((0, 1), dtype=torch.int8)
        
        # signal indices
        self.signal_indices: list[list[int]] = []

        self.agent_indices: list[list[int]] = []

        self.agent_masks = torch.zeros((0,), dtype=torch.bool)
        self.trajectory_masks = torch.zeros((0, self.config.seq_len), dtype=torch.bool)

    #region dataframe post-process
    @staticmethod
    def postprocess_pedestrian_traj(df: pd.DataFrame) -> pd.DataFrame:
        df['yaw_rad'] = np.arctan2(df['ay'], df['ax'])
        df['heading_rad'] = df['yaw_rad']

        df['v_lat'] = 0.0
        df['v_lon'] = df['vx'] / np.cos(df['heading_rad'])
        
        df['a_lat'] = 0.0
        df['a_lon'] = df['ax'] / np.cos(df['heading_rad'])

        return df

    @staticmethod
    def postprocess_pedestrian_meta(df: pd.DataFrame) -> pd.DataFrame:
        df['length'] = pedestrian_length
        df['width'] = pedestrian_width
        df['CrossType'] = CrossType.NoRecord
        df['Signal_Violation_Behavior'] = SignalViolationBehavior.no_record
        df['agent_type'] = AgentType.pedestrian

        return df
    #endregion

    def load_records(
        self,
        dir: str, 
        record_names: list[str],
        verbose: bool = True
    ):
        '''
        read one record

        Args:
        ---
        - dir: dataset folder
        - record_name: record name
        - verbose
        '''
        agent_idx_offset = self.agent_records.shape[0]

        trajectory_idx_offset = self.trajectory_records.shape[0]
        signal_idx_offset = self.signal_records.shape[0]

        trajectory_records: list[list[list[float]]] = []
        trajectory_masks: list[torch.BoolTensor] = []

        trajectory_indices = []


        signal_indices: list[list[int]] = []

        agent_indices: list[list[int]] = []


        signal_records: list[torch.IntTensor] = []

        agent_records: list[torch.FloatTensor] = []
        agent_flags: list[torch.IntTensor] = []
        agent_masks: list[torch.BoolTensor] = []
        

        for record_name in tqdm.tqdm(record_names, desc='load_records', disable=not verbose):
            record_path = f'{dir}/{record_name}'
            
            #read agent meta
            agent_df = pd.concat((
                io.read_vehicle_meta(f'{record_path}/Veh_tracks_meta.csv'),
                SignalizedIntersectionDataset.postprocess_pedestrian_meta(io.read_pedestrian_meta(f'{record_path}/Ped_tracks_meta.csv'))
            ), ignore_index=True)

            agent_df = agent_df.drop_duplicates(subset=['track_id'])


            # build index
            agent_index: dict[str, int] = { track_id: index + agent_idx_offset for index, track_id in enumerate(agent_df['track_id']) }
            agent_idx_offset += len(agent_index)

            # read traffic signals
            signal_df = io.read_traffic_signals(f'{record_path}/TrafficLight_{get_traffic_signal_record_name(record_name)}.csv')
            signal_df = signal_df.drop_duplicates(subset=['frame_id'])

            # read trajectories
            trajectory_df = pd.concat((
                io.read_vehicle_trajectories(f'{record_path}/Veh_smoothed_tracks.csv'),
                SignalizedIntersectionDataset.postprocess_pedestrian_traj(io.read_pedestrian_trajectories(f'{record_path}/Ped_smoothed_tracks.csv'))
            ), ignore_index=True)

            trajectory_df = trajectory_df.drop_duplicates(subset=['track_id', 'frame_id'])
            trajectory_df = trajectory_df[(trajectory_df['frame_id'] % self.config.stride) == 0]
            trajectory_df.sort_values(by=['frame_id', 'track_id'], ascending=True, inplace=True, ignore_index=True)

            base_signal_df_pointer = 0
            base_trajectory_df_pointer = 0


            # build tensors
            for current_frame in range(trajectory_df['frame_id'].min(), trajectory_df['frame_id'].max() - self.config.stride * self.config.seq_len + 1, self.config.stride):
                # step trajectory pointer
                while trajectory_df.at[base_trajectory_df_pointer, 'frame_id'] < current_frame:
                    base_trajectory_df_pointer += 1

                trajectory_df_pointer = base_trajectory_df_pointer


                # step signal pointer
                if base_signal_df_pointer < len(signal_df) and current_frame >= signal_df.at[base_signal_df_pointer + 1, 'frame_id']:
                    base_signal_df_pointer += 1

                signal_df_pointer = base_signal_df_pointer


                # get signal states
                signal_states = [-1] * self.config.seq_len

                for i, offset in enumerate(range(self.config.seq_len)):
                    if signal_df_pointer < len(signal_df) and current_frame + offset * self.config.stride >= signal_df.at[signal_df_pointer + 1, 'frame_id']:
                        signal_df_pointer += 1

                    signal_states[i] = signal_df_pointer + signal_idx_offset
                

                trajectory_dict: dict[str, list[int]] = dict()

                # read frames
                while trajectory_df_pointer < len(trajectory_df):
                    frame_id: int = trajectory_df.at[trajectory_df_pointer, 'frame_id']

                    if frame_id >= current_frame + self.config.seq_len * self.config.stride:
                        break

                    agent_id = trajectory_df.at[trajectory_df_pointer, 'track_id']

                    if agent_id not in trajectory_dict:
                        trajectory_dict[agent_id] = [-1] * self.config.seq_len

                    trajectory_dict[agent_id][(frame_id - current_frame) // self.config.stride] = trajectory_df_pointer

                    trajectory_df_pointer += 1

                
                current_agent_ids = sorted((agent_id for agent_id in trajectory_dict.keys()), key=lambda agent_id: agent_index[agent_id])
                local_agent_index = { agent_id: index for index, agent_id in enumerate(current_agent_ids) }
                
                # build trajectories
                current_trajectory_records = np.full((len(trajectory_dict), self.config.seq_len, TRAJECTORY_FEATURE_DIM), self.config.padding_value, dtype=np.float32)

                current_trajectory_mask = np.zeros((len(trajectory_dict), self.config.seq_len), dtype=bool)
                current_agent_mask = np.zeros(len(trajectory_dict), dtype=bool)

                # build trajectories
                for i, (agent_id, selector) in enumerate(trajectory_dict.items()):
                    selector = np.array(selector, dtype=int)
                    selector_mask = selector != -1

                    agent_idx = local_agent_index[agent_id]

                    current_agent_mask[agent_idx] = np.bitwise_and(
                        np.any(selector_mask[:self.config.obs_len], axis=-1),
                        np.any(selector_mask[self.config.obs_len:], axis=-1)
                    )
                    
                    current_trajectory_mask[agent_idx, :] = selector_mask
                    current_trajectory_records[agent_idx, selector_mask, :] = trajectory_df.loc[selector[selector_mask], TRAJECTORY_FEATURES].to_numpy(dtype=np.float32)
                    
                if not self.config.include_incomplete_trajectories:
                    # remove agent not present in the entire window
                    current_agent_mask = current_trajectory_mask.all(axis=-1)
                    current_trajectory_mask[~current_agent_mask] = False

                if current_agent_mask.sum() < self.config.min_predictions:
                    continue
                
                # accumulate
                trajectory_records.append(torch.from_numpy(current_trajectory_records))

                signal_indices.append(signal_states)
                agent_indices.append([agent_index[agent_id] for agent_id in current_agent_ids])

                trajectory_indices.append((trajectory_idx_offset, trajectory_idx_offset + len(current_agent_ids)))
                trajectory_idx_offset += len(trajectory_dict)

                trajectory_masks.append(torch.from_numpy(current_trajectory_mask))
                agent_masks.append(torch.from_numpy(current_agent_mask))
                
            # concat data
            signal_records.append(torch.from_numpy(signal_df[SIGNAL_FEATURES].to_numpy(dtype=np.int8)))

            if self.config.encode_traffic_signals:
                signal_records[-1] = encode_traffic_signals(signal_records[-1])

            signal_idx_offset += len(signal_records)

            
            agent_records.append(torch.from_numpy(agent_df[AGENT_FEATURES].to_numpy(dtype=np.float32)))
            agent_flags.append(torch.from_numpy(agent_df[AGENT_FLAGS].to_numpy(dtype=np.int8)))


        # apply changes
        self.signal_records = torch.concat((self.signal_records, *signal_records))
        self.signal_indices += signal_indices

        self.agent_records = torch.concat((self.agent_records, *agent_records))
        self.agent_flags = torch.concat((self.agent_flags, *agent_flags))
        
        self.trajectory_records = torch.concat((self.trajectory_records, *trajectory_records))

        self.agent_indices += agent_indices
        self.trajectory_indices += trajectory_indices

        self.trajectory_masks = torch.concat((self.trajectory_masks, *trajectory_masks))
        self.agent_masks = torch.concat((self.agent_masks, *agent_masks))


    #region cache methods
    def save_cache(self, cache_path: str):
        torch.save({
            'config': self.config.serialize(),
            'trajectory_records': self.trajectory_records,
            'agent_records': self.agent_records,
            'agent_flags': self.agent_flags,
            'signal_records': self.signal_records,
            'signal_indices': self.signal_indices,
            'agent_indices': self.agent_indices,
            'trajectory_indices': self.trajectory_indices,
            'agent_masks': self.agent_masks,
            'trajectory_masks': self.trajectory_masks
        }, cache_path)

    @classmethod
    def load_cache(cls, cache_path: str):
        cache_dict = torch.load(cache_path)

        dataset = cls(
            SignalizedIntersectionDatasetConfig.deserialize(cache_dict['config'])
        )

        dataset.trajectory_records = cache_dict['trajectory_records']
        
        dataset.trajectory_masks = cache_dict['trajectory_masks']
        dataset.agent_masks = cache_dict['agent_masks']

        dataset.agent_records = cache_dict['agent_records']
        dataset.agent_flags = cache_dict['agent_flags']

        dataset.signal_records = cache_dict['signal_records']
        dataset.signal_indices = cache_dict['signal_indices']

        dataset.agent_indices = cache_dict['agent_indices']

        dataset.trajectory_indices = cache_dict['trajectory_indices']

        return dataset
    #endregion

    #region torch methods
    def __len__(self):
        return len(self.agent_indices)

    def __getitem__(self, idx: int, remove_future_agents: bool = True):
        '''
        get item

        Args:
        ---
        - idx: index
        - remove_future_agents: remove agents not appearing in `0:config.obs_len`

        Returns:
        ---
        - trajectory: [A, T, TRAJ_DIM]
        - trajectory_mask: [A, T]
        - agent_mask: [A]
        - agent_records: [A, AGENT_DIM]
        - agent_flags: [A, FLAG_DIM]
        - signals: signal records [A, SIG_DIM]
        '''
        traj_start, traj_end = self.trajectory_indices[idx]

        trajectory = self.trajectory_records[traj_start:traj_end]


        signals = self.signal_records[self.signal_indices[idx]]

        trajectory_mask = self.trajectory_masks[traj_start:traj_end]
        agent_mask = self.agent_masks[traj_start:traj_end]

        agent_indices = self.agent_indices[idx]
        agent_records = self.agent_records[agent_indices]
        agent_flags = self.agent_flags[agent_indices]

        if remove_future_agents:
            agent_records = agent_records[agent_mask]
            agent_flags = agent_flags[agent_mask]

            trajectory = trajectory[agent_mask]
            trajectory_mask = trajectory_mask[agent_mask]

            agent_mask = agent_mask[agent_mask]

        return (
            trajectory,
            trajectory_mask,
            agent_mask,
            agent_records,
            agent_flags,
            signals
        )

    @staticmethod
    def collate_nested(batches) -> tuple[
        torch.FloatTensor, 
        torch.BoolTensor, 
        torch.BoolTensor, 
        torch.FloatTensor, 
        torch.IntTensor, 
        torch.IntTensor
    ]:
        '''
        collate the batch using the experimental `nested_tensor` api

        Args:
        ---
        - batches: batch list list[B]

        Returns:
        ---
        - trajectory: [B, A[i], T, TRAJ_DIM]
        - trajectory_mask: [B, A[i], T]
        - agent_mask: [B, A[i]]
        - agent_records: [B, A[i], AGENT_DIM]
        - agent_flags: [B, A[i], FLAG_DIM]
        - signals: signal records [B, A[i], SIG_DIM]
        '''
        return tuple([
            torch.nested.nested_tensor([batch[i] for batch in batches])
            for i in range(6)
        ])
    #endregion