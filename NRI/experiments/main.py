import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from NRI.models import DynamicNeuralRelationalInference

from NRI.dataset import SignalizedIntersectionDatasetForNRI
from NRI.experiments.train import train_one_epoch
from NRI.experiments.valid import valid_one_epoch
from NRI.experiments.config import ExperimentConfig

def checkpoint_model(
    model: DynamicNeuralRelationalInference,
    path: str,
    stats: dict[str, any]
):
    torch.save({
        'params': model.state_dict(),
        'stats': stats or dict()
    }, f'{path}.pt')

def train(
    model: DynamicNeuralRelationalInference,
    train_set: SignalizedIntersectionDatasetForNRI,
    valid_set: SignalizedIntersectionDatasetForNRI,
    config: ExperimentConfig,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler
):
    '''
    train model for one epoch
    '''
    train_loader = DataLoader(
        train_set, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=SignalizedIntersectionDatasetForNRI.collate_padded if config.batched_training else SignalizedIntersectionDatasetForNRI.collate_nested,
        num_workers=4
    )
    
    valid_loader = DataLoader(
        valid_set, 
        batch_size=config.batch_size, 
        shuffle=False, 
        collate_fn=SignalizedIntersectionDatasetForNRI.collate_padded,
        num_workers=4
    )

    best_ade: float = float('inf')

    ade: list[float] = []
    fde: list[float] = []
    signal_acc: list[float] = []
    
    overall_loss: list[float] = []
    encoder_loss: list[float] = []
    decoder_loss: list[float] = []
    signal_loss: list[float] = []

    for epoch in range(config.n_epoch):
        train_stats = train_one_epoch(
            model, 
            train_loader, 
            optimizer, 
            tqdm_desc=f'[train] epoch {epoch}',
            config=config
        )

        train_stats.print()

        valid_stats = valid_one_epoch(
            model,
            valid_loader,
            tqdm_desc=f'[valid] epoch {epoch}',
            config=config
        )

        valid_stats.print()

        lr_scheduler.step()

        overall_loss.append(train_stats.loss)
        encoder_loss.append(train_stats.encoder_loss)
        decoder_loss.append(train_stats.decoder_loss)
        signal_loss.append(train_stats.signal_loss)

        fde.append(valid_stats.fde)
        ade.append(valid_stats.ade)
        signal_acc.append(valid_stats.signal_prediction_acc)

        if epoch > 0 and epoch % config.checkpoint_interval == 0:
            # checkpoint
            checkpoint_model(model, f'{config.checkpoint_prefix}_ep{epoch}', {
                'ade': ade,
                'fde': fde,
                'signal_acc': signal_acc,
                'overall_loss': overall_loss,
                'encoder_loss': encoder_loss,
                'decoder_loss': decoder_loss,
                'signal_loss': signal_loss
            })

        if valid_stats.ade < best_ade:
            best_ade = valid_stats.ade
            
            checkpoint_model(model, f'{config.checkpoint_prefix}_best', {
                'ade': ade,
                'fde': fde,
                'signal_acc': signal_acc,
                'overall_loss': overall_loss,
                'encoder_loss': encoder_loss,
                'decoder_loss': decoder_loss,
                'signal_loss': signal_loss
            })

    checkpoint_model(model, f'{config.checkpoint_prefix}_final', {
        'ade': ade,
        'fde': fde,
        'signal_acc': signal_acc,
        'overall_loss': overall_loss,
        'encoder_loss': encoder_loss,
        'decoder_loss': decoder_loss,
        'signal_loss': signal_loss
    })
