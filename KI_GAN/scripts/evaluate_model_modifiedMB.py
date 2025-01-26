import argparse
import os
import torch
import logging
import collections
import collections.abc

for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))

from attrdict import AttrDict
from kigan.data.loader import data_loader
from kigan.models import TrajectoryGenerator
from kigan.losses import displacement_error, final_displacement_error
from kigan.utils import relative_to_abs, get_dset_path

parser = argparse.ArgumentParser()

model_path_ = 'checkpoint_with_model_.pt'
parser.add_argument('--model_path', default=model_path_, type=str)
parser.add_argument('--num_samples', default=30, type=int)
parser.add_argument('--dset_type', default='val', type=str)
parser.add_argument('--log_file', default='evaluation.log', type=str)


def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator

# This is calculating minimum average
# Change order of sum and mean
# Calculate min first and sum together
# Store all results into one file and then directly do it - just once
def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def evaluate(args, loader, generator, num_samples):
    ade_outer, fde_outer = [], []
    total_traj = 0
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
             loss_mask, seq_start_end, vx, vy, ax, ay, agent_type, size, traffic_state) = batch
            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)

            for _ in range(num_samples):
                pred_traj_fake_rel = generator(
                    obs_traj, obs_traj_rel, seq_start_end,vx, vy, ax, ay, agent_type, size, traffic_state
                )
                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel, obs_traj[-1]
                )
                ade.append(displacement_error(
                    pred_traj_fake, pred_traj_gt, mode='raw'
                ))
                fde.append(final_displacement_error(
                    pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
                ))
            # Change 
            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)
        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / (total_traj)
        return ade, fde


def main(args):
    logging.basicConfig(filename=args.log_file, level=logging.INFO)
    scenarios = [
        {

            'description': 'Baseline model with full test set',
            'model_path': '/home/mattb/AnythingVisionaries/KI_GAN/scripts/Tianjin_Baseline_Training/checkpoint_with_model_74.pt',
            'dataset_path': '/home/mattb/AnythingVisionaries/KI_GAN/datasets/Tianjin/test'
        },
        {
            'description': 'No traffic encoder model with no traffic encoder test set',
            'model_path': '/home/mattb/AnythingVisionaries/KI_GAN/scripts/Tianjin_NoTraffic_Training/checkpoint_with_model_74.pt',
            'dataset_path': '/home/mattb/AnythingVisionaries/KI_GAN/datasets/Tianjin_2/test'
        },
    ]

    for scenario in scenarios:
        args.model_path = scenario['model_path']
        dataset_path = scenario['dataset_path']
        description = scenario['description']

        logging.info(f"Evaluating scenario: {description}")
        if os.path.isdir(args.model_path):
            filenames = os.listdir(args.model_path)
            filenames.sort()
            paths = [
                os.path.join(args.model_path, file_) for file_ in filenames
            ]
        else:
            paths = [args.model_path]

        for path in paths:
            checkpoint = torch.load(path)
            generator = get_generator(checkpoint)
            _args = AttrDict(checkpoint['args'])
            logging.info(_args)
            _, loader = data_loader(_args, dataset_path)
            ade, fde = evaluate(_args, loader, generator, args.num_samples)
            logging.info(f'Scenario: {description}, Dataset: {_args.dataset_name}, Pred Len: {_args.pred_len}, ADE: {ade:.2f}, FDE: {fde:.2f}')
            logging.info('-' * 80)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
