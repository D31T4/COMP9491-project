from SinD.vis.map import IntersectionMap
from SinD.utils.geom import calculate_rot_bboxes_and_triangle
from SinD.vis.scene_rasterizer import SceneRasterizer

from SinD.dataset import SignalizedIntersectionDataset
from SinD.dataset.type import AgentType, SignalViolationBehavior

import numpy as np
import matplotlib.pyplot as plt

class DatasetRasterizer(SceneRasterizer):
    '''
    Visualize dataset
    '''
    
    def __init__(
        self, 
        dataset: SignalizedIntersectionDataset, 
        map: IntersectionMap, 
        readonly_slider: bool = False
    ):
        '''
        Args:
        ---
        - dataset
        - map: intersection map
        - readonly_slider: allow dragging slider

        Example:
        ---
        ```
        >>> rasterizer = DatasetRasterizer(dataset, map)
        ```
        '''
        self.dataset = dataset
        self.map = map

        self.num_frames = len(dataset) + dataset.config.seq_len - 1

        self.init_params()
        self.init_ui(readonly_slider)

    @property
    def config(self):
        return self.dataset.config
    
    def get_trajectory_data(self):
        '''
        get trajectory data at current frame

        Returns:
        ---
        - trajectory
        - trajectory_mask
        - agents
        - agent_flags
        - agent_ids: unique agent identifier
        - future_trajectory
        - future_trajectory_mask
        '''
        dataset_len = len(self.dataset)

        #region get current loc, agent
        offset = 0
        index = self.current_frame

        # get last seq_len frames
        if self.current_frame >= dataset_len:
            offset = self.current_frame - dataset_len + 1
            index = -1

        agent_ids = self.dataset.agent_indices[index]
        traj_start, traj_end = self.dataset.trajectory_indices[index]

        agents = self.dataset.agent_records[agent_ids]
        agent_flags = self.dataset.agent_flags[agent_ids]

        trajectory_mask = self.dataset.trajectory_masks[traj_start:traj_end, offset]
        trajectory = self.dataset.trajectory_records[traj_start:traj_end, offset]


        agent_ids = [agent_id for i, agent_id in enumerate(agent_ids) if trajectory_mask[i].item()]

        trajectory = trajectory[trajectory_mask]
        agents = agents[trajectory_mask]
        agent_flags = agent_flags[trajectory_mask]
        #endregion

        #region get future trajectory
        future_trajectory = None
        future_trajectory_mask = None
        future_agent_ids = None


        if (future_index := self.current_frame - self.config.obs_len) >= 0:
            offset = 0

            # get future trajectories of agents still active in current frame
            if future_index >= dataset_len:
                offset = future_index - dataset_len + 1
                future_index = -1

            traj_start, traj_end = self.dataset.trajectory_indices[future_index]

            future_agent_ids = self.dataset.agent_indices[future_index]
            future_agent_mask = self.dataset.agent_masks[traj_start:traj_end]
            
            future_trajectory = self.dataset.trajectory_records[traj_start:traj_end, (self.config.obs_len + offset):, :2]
            future_trajectory_mask = self.dataset.trajectory_masks[traj_start:traj_end, (self.config.obs_len + offset):]

            # remove inactive agents
            if offset != 0:
                future_agent_mask = future_agent_mask & trajectory_mask

            future_trajectory = future_trajectory[future_agent_mask]
            future_trajectory_mask = future_trajectory_mask[future_agent_mask]
            future_agent_ids = [agent_id for i, agent_id in enumerate(future_agent_ids) if future_agent_mask[i].item()]
        #endregion

        return (
            trajectory, 
            agents, 
            agent_flags, 
            agent_ids, 
            future_trajectory, 
            future_trajectory_mask,
            future_agent_ids
        )

    def gc(self, active_ids: list[int]):
        '''
        remove out-of-frame agents

        Args:
        ---
        - active_ids: list of agents in frame
        '''
        to_gc: list[int] = []

        for id, entities in self.entities['agents'].items():
            if id in active_ids:
                if self.current_frame < self.config.obs_len:
                    ref = entities.pop('trajectory', None)
                    ref and ref.remove()

                continue

            for entity in entities.values():
                entity.remove()

            to_gc.append(id)

        for id in to_gc:
            del self.entities['agents'][id]

    def update_agents(self):
        '''
        update agent positions
        '''
        (
            trajectory, 
            agents, 
            agent_flags, 
            agent_ids, 
            future_trajectory, 
            future_trajectory_mask,
            future_agent_id
        ) = self.get_trajectory_data()

        mask = agent_flags[:, 0] != AgentType.pedestrian # pedestrian dont have length, width attributes
        
        bboxes, triangles = calculate_rot_bboxes_and_triangle(
            trajectory[mask, :2],
            agents[mask, 0],
            agents[mask, 1],
            trajectory[mask, 6]
        )

        bboxes = bboxes.numpy()
        triangles = triangles.numpy()

        bbox_pointer = 0

        self.gc(agent_ids)

        for i in range(trajectory.size(0)):
            agent_id = agent_ids[i]

            agent_type = agent_flags[i, 0].item()
            color = self.colors[agent_type]
            x, y = trajectory[i, :2].tolist()


            match agent_flags[i, 2].item():
                case SignalViolationBehavior.red_light_running:
                    text_color = 'red'
                case SignalViolationBehavior.yellow_light_running:
                    text_color = 'yellow'
                case SignalViolationBehavior.no_violation:
                    text_color = 'green'
                case _:
                    text_color = 'black'

            entity_ref = self.entities['agents'].get(agent_id, None)

            if agent_type == AgentType.pedestrian:
                # render pedestrian
                if entity_ref is None:
                    circle = plt.Circle((x, y), radius=0.5, zorder=20, color=color, fill=True)
                    self.ax.add_patch(circle)
                    
                    text = self.ax.text(x, y + 2, str(agent_id), horizontalalignment='center', zorder=30)

                    entity_ref = { "circle": circle, "text": text }
                    self.entities["agents"][agent_id] = entity_ref
                else:
                    entity_ref["circle"].set_center((x, y))
                    entity_ref["text"].set_position((x, y + 2))

            else:
                # render vehicle
                bbox: np.ndarray = bboxes[bbox_pointer]
                triangle: np.ndarray = triangles[bbox_pointer]
                bbox_pointer += 1

                if entity_ref is None:
                    rect = plt.Polygon(bbox, closed=True, zorder=20, color=color, fill=True, alpha=0.6)
                    triangle = plt.Polygon(triangle, closed=True, facecolor="k", fill=True, edgecolor="k", lw=0.1, alpha=0.6, zorder=21)

                    self.ax.add_patch(rect)
                    self.ax.add_patch(triangle)

                    text = self.ax.text(x, y + 1.5, str(agent_id), horizontalalignment='center', zorder=30, color=text_color)
                
                    entity_ref = { "rect": rect, "tri": triangle, "text": text }
                    self.entities["agents"][agent_id] = entity_ref
                else:
                    entity_ref["rect"].set_xy(bbox)
                    entity_ref["tri"].set_xy(triangle)
                    entity_ref["text"].set_position((x, y + 1.5))
        

        # plot future trajectory
        if future_trajectory is None:
            return

        for i in range(future_trajectory.size(0)):
            current_future_trajectory = future_trajectory[i, future_trajectory_mask[i]].numpy()

            agent_id = future_agent_id[i]
            entity_ref = self.entities['agents'][agent_id]

            if (trajectory_ref := entity_ref.get('trajectory', None)) is None:
                entity_ref['trajectory'] = self.ax.plot(
                    current_future_trajectory[:, 0], 
                    current_future_trajectory[:, 1], 
                    color="green", 
                    linewidth=1, 
                    alpha=0.7, 
                    zorder=10
                )[0]
            else:
                trajectory_ref.set_data(current_future_trajectory[:, 0], current_future_trajectory[:, 1])
    
    def get_signals_data(self):
        dataset_len = len(self.dataset)

        if self.current_frame < dataset_len:
            index = self.dataset.signal_indices[self.current_frame][0]
        else:
            index = self.dataset.signal_indices[-1][self.current_frame - dataset_len + 1]
        
        return self.dataset.signal_records[index]
