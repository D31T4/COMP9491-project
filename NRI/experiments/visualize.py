import torch

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from SinD.dataset import SignalizedIntersectionDatasetConfig

from SinD.dataset.type import AgentType
from SinD.vis.map import IntersectionMap
from SinD.vis.scene_rasterizer import SceneRasterizer


def plot_fde(fde: list[float]):
    plt.figure()
    ax = plt.subplot()
    epochs = np.arange(len(fde))

    ax.plot(epochs, fde)

    ax.set_ylabel('FDE (m)')
    ax.set_xlabel('Epoch')

    return ax

def plot_ade(ade: list[float]):
    plt.figure()
    ax = plt.subplot()
    epochs = np.arange(len(ade))

    ax.plot(epochs, ade)

    ax.set_ylabel('ADE (m)')
    ax.set_xlabel('Epoch')

    return ax

def plot_de_by_step(de: list[float]):
    plt.figure()
    ax = plt.subplot()
    timesteps = np.arange(len(de))

    ax.plot(timesteps, de)

    ax.set_ylabel('DE (m)')
    ax.set_xlabel('Timestep')

    return ax

def plot_de_by_class(
    ade: dict[AgentType, float], 
    fde: dict[AgentType, float]
):
    plt.figure()
    ax = plt.subplot()

    bar_loc = np.arange(len(AgentType))
    bar_width = 0.25

    ax.bar(bar_loc, [ade[int(agent_type)] for agent_type in AgentType], label='ADE', width=bar_width)
    ax.bar(bar_loc + bar_width, [fde[int(agent_type)] for agent_type in AgentType], label='FDE', width=bar_width)
    ax.set_xticks(bar_loc + bar_width / 2, [agent_type.name for agent_type in AgentType])
    ax.legend()

    return ax


class StaticPredictionVisualizer(SceneRasterizer):
    def __init__(
        self,
        trajectory: torch.FloatTensor,
        trajectory_mask: torch.BoolTensor,
        agents: torch.FloatTensor,
        agent_flags: torch.IntTensor,
        signals: torch.IntTensor,
        map: IntersectionMap,
        config: SignalizedIntersectionDatasetConfig,
        agent_ids: list[int] | None = None
    ):
        self.map = map

        self.config = config
        
        self.trajectory = trajectory
        self.trajectory_mask = trajectory_mask
        self.agents = agents
        self.agent_flags = agent_flags
        self.signals = signals
        self.agent_ids = agent_ids or [*range(agents.size(0))]

        self.init_params()
        self.init_ui()

    def __del__(self):
        pass

    def init_params(self):
        super().init_params()
        self.current_frame = self.config.obs_len

    def init_ui(self):
        self.ax = self.map.plot()
        self.update_agents()
        self.update_signals()

    

def plot_predicted_trajectories(
    ground_truth: torch.FloatTensor,
    predictions: torch.FloatTensor,
    trajectory_mask: torch.BoolTensor,
    agent_class: torch.IntTensor,
    agent_feats: torch.FloatTensor,
    signals: torch.IntTensor,
    map: IntersectionMap,
    config: SignalizedIntersectionDatasetConfig
):
    visualizer = StaticPredictionVisualizer(
        ground_truth,
        trajectory_mask,
        agent_feats,
        agent_class,
        signals,
        map,
        config
    )

    ground_truth = ground_truth.numpy()
    predictions = predictions.numpy()
    trajectory_mask = trajectory_mask.numpy()

    for agent_index in range(trajectory_mask.shape[0]):
        if not trajectory_mask[agent_index].all(axis=-1):
            continue

        visualizer.ax.plot(
            ground_truth[agent_index, :, 0], 
            ground_truth[agent_index, :, 1], 
            'k-', 
            alpha=0.8,
            zorder=19
        )

        visualizer.ax.arrow(
            ground_truth[agent_index, -2, 0],
            ground_truth[agent_index, -2, 1],
            ground_truth[agent_index, -1, 0] - ground_truth[agent_index, -2, 0],
            ground_truth[agent_index, -1, 1] - ground_truth[agent_index, -2, 1],
            width=0.5,
            zorder=19,
            color='black'
        )
        
        visualizer.ax.plot(
            predictions[agent_index, :, 0], 
            predictions[agent_index, :, 1], 
            'g-', 
            marker='+', 
            alpha=0.8,
            zorder=19
        )

    return visualizer.ax

def plot_kigan(
    ground_truth: torch.FloatTensor,
    predictions: torch.FloatTensor,
    map: IntersectionMap,
    config: SignalizedIntersectionDatasetConfig
):
    ax = map.plot()

    ground_truth_cmap = plt.get_cmap('Reds')
    prediction_cmap = plt.get_cmap('Greens')

    for agent_index in range(ground_truth.size(0)):
        for timestamp in range(config.seq_len - 1):
            ax.plot(
                ground_truth[agent_index, timestamp:(timestamp + 2), 0], 
                ground_truth[agent_index, timestamp:(timestamp + 2), 1], 
                color=ground_truth_cmap(float(timestamp) / config.seq_len), 
                marker='o',
                zorder=19
            )

        for timestamp in range(config.pred_len - 1):
            ax.plot(
                predictions[agent_index, timestamp:(timestamp + 2), 0], 
                predictions[agent_index, timestamp:(timestamp + 2), 1], 
                color=prediction_cmap(float(timestamp) / config.pred_len), 
                marker='o',
                zorder=19
            )

    red_line = matplotlib.lines.Line2D([], [], color='red', marker='_', markersize=15, label='Ground Truth')
    green_line = matplotlib.lines.Line2D([], [], color='green', marker='_', markersize=15, label='Predicted')

    ax.legend(handles=[red_line, green_line])

    return ax