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
from kigan.utils import relative_to_abs

parser = argparse.ArgumentParser()

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
                    obs_traj, obs_traj_rel, seq_start_end, vx, vy, ax, ay, agent_type, size, traffic_state
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

            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)
        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / (total_traj)
        return ade, fde

def evaluate_and_save_results(model_path, dataset_path, num_samples, scenario_description, output_dir):
    checkpoint = torch.load(model_path)
    generator = get_generator(checkpoint)
    _args = AttrDict(checkpoint['args'])
    _, loader = data_loader(_args, dataset_path)
    ade, fde = evaluate(_args, loader, generator, num_samples)

    # Log results
    logging.info(
        f'Model Path: {model_path}, Dataset Path: {dataset_path}, Pred Len: {_args.pred_len}, ADE: {ade:.2f}, FDE: {fde:.2f}')
    logging.info('-' * 80)

    # Save results to a text file
    results_file = os.path.join(output_dir, f'{scenario_description}_results.txt')
    with open(results_file, 'w') as f:
        f.write(f'Model Path: {model_path}\n')
        f.write(f'Dataset Path: {dataset_path}\n')
        f.write(f'Prediction Length: {_args.pred_len}\n')
        f.write(f'Average Displacement Error (ADE): {ade:.2f}\n')
        f.write(f'Final Displacement Error (FDE): {fde:.2f}\n')

def main(args):
    logging.basicConfig(filename=args.log_file, level=logging.INFO)

    scenarios = [
        {
            'description': '12 Step Baseline Model',
            'model_path': '/home/mattb/AnythingVisionaries/KI_GAN/scripts/Model_Baseline_12_UniformNoise/checkpoint_with_model_74.pt',
            'dataset_path': '/home/mattb/AnythingVisionaries/KI_GAN/datasets/Tianjin/test'
        },
        {
            'description': '18 Step Baseline Model',
            'model_path': '/home/mattb/AnythingVisionaries/KI_GAN/scripts/Model_Baseline_18_UniformNoise_Bs64/checkpoint_with_model_75.pt',
            'dataset_path': '/home/mattb/AnythingVisionaries/KI_GAN/datasets/Tianjin/test'
        },
        {
            'description': '24 Step Baseline Model',
            'model_path': '/home/mattb/AnythingVisionaries/KI_GAN/scripts/Model_Baseline_24_UniformNoise_Bs64/checkpoint_with_model_75.pt',
            'dataset_path': '/home/mattb/AnythingVisionaries/KI_GAN/datasets/Tianjin/test'
        },
        {
            'description': '12 Step No Traffic Encoder Model',
            'model_path': '/home/mattb/AnythingVisionaries/KI_GAN/scripts/Model_NoTraff_12_UniformNoise_bs64/checkpoint_with_model_74.pt',
            'dataset_path': '/home/mattb/AnythingVisionaries/KI_GAN/datasets/Tianjin_2/test'
        },
        {
            'description': '18 Step No Traffic Encoder Model',
            'model_path': '/home/mattb/AnythingVisionaries/KI_GAN/scripts/Model_NoTraff_18_UniformNoise_bs64/checkpoint_with_model_75.pt',
            'dataset_path': '/home/mattb/AnythingVisionaries/KI_GAN/datasets/Tianjin_2/test'
        },
        {
            'description': '24 Step No Traffic Encoder Model',
            'model_path': '/home/mattb/AnythingVisionaries/KI_GAN/scripts/Model_NoTraff_24_UniformNoise_bs64/checkpoint_with_model_75.pt',
            'dataset_path': '/home/mattb/AnythingVisionaries/KI_GAN/datasets/Tianjin_2/test'
        },
    ]

    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    for scenario in scenarios:
        model_path = scenario['model_path']
        dataset_path = scenario['dataset_path']

        logging.info(f"Evaluating scenario: {scenario['description']}")
        evaluate_and_save_results(
            model_path,
            dataset_path,
            args.num_samples,
            scenario['description'],
            output_dir
        )

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
