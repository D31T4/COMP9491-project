from typing import Literal, Optional
from pathlib import Path
import os

DEFAULT_CHECKPOINT_PATH = f'{Path(__file__).parent.parent}/checkpoints'

class ExperimentConfig:
    '''
    experiemnt configuration
    '''

    def __init__(
        self,
        obs_len: int,
        pred_len: int,
        encoder_loss_weight: float,
        decoder_loss_weight: float,
        signal_loss_weight: float,
        n_epoch: int,
        batch_size: int,
        checkpoint_prefix: str,
        include_burn_in_loss: bool = False,
        checkpoint_interval: int = float('inf'),
        use_cuda: bool = False,
        avg_logits: Optional[Literal['avg', 'ema']] = None,
        ema_alpha: Optional[float] = None,
        checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
        batched_training: bool = True,
        mask_traffic_signal: bool = False,
        teacher_forcing_steps: int = float('inf')
    ):
        '''
        Args:
        ---
        - obs_len: observation length
        - pred_len: prediction length
        - encoder_loss_weight: encoder loss weight
        - decoder_loss_weight: decoder loss weight
        - signal_loss_weight: signal loss weight
        - n_epoch: no. epochs
        - batch_size: batch size
        - checkpoint_prefix: prefix of checkpoint save file name
        - include_burn_in_loss: include decoder burn-in loss
        - checkpoint_interval: checkpoint every n epochs
        - use_cuda: use cuda
        - avg_logits
        - ema_alpha
        - checkpoint_path: directory to save checkpoints
        - batched_training: use batch training loss
        - mask_traffic_signal: mask traffic signals
        '''
        self.obs_len = obs_len
        self.pred_len = pred_len

        self.n_epoch = n_epoch
        self.batch_size = batch_size

        self.encoder_loss_weight = encoder_loss_weight
        self.decoder_loss_weight = decoder_loss_weight
        self.signal_loss_weight = signal_loss_weight
        self.include_burn_in_loss = include_burn_in_loss

        # create checkpoint dir if not exist
        os.makedirs(checkpoint_path, exist_ok=True)

        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_prefix = f'{checkpoint_path}/{checkpoint_prefix}'

        self.use_cuda = use_cuda

        self.avg_logits = avg_logits
        self.ema_alpha = ema_alpha

        self.batched_training = batched_training

        self.mask_traffic_signal = mask_traffic_signal

        self.teacher_forcing_steps = teacher_forcing_steps

        if self.mask_traffic_signal:
            self.signal_loss_weight = 0.0