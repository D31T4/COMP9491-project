import matplotlib
import matplotlib.backend_bases
import matplotlib.patches
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

import matplotlib.widgets
import torch
import numpy as np

from SinD.vis.map import IntersectionMap
from SinD.dataset.dataset import SignalizedIntersectionDatasetConfig, decode_traffic_signals
from SinD.dataset.type import AgentType, TrafficSignalType, SignalViolationBehavior
from SinD.utils.geom import calculate_rot_bboxes_and_triangle

# drone fps
DRONE_FRAMERATE = 29.97

class SceneRasterizer:
    '''
    Visualize scene
    '''

    def __init__(
        self, 
        trajectory: torch.FloatTensor,
        trajectory_mask: torch.BoolTensor,
        agents: torch.FloatTensor,
        agent_flags: torch.IntTensor,
        signals: torch.IntTensor,
        map: IntersectionMap, 
        config: SignalizedIntersectionDatasetConfig,
        agent_ids: list[int] | None = None,
        readonly_slider: bool = False
    ):
        '''
        Args
        ---
        - trajectory
        - trajectory_mask
        - agents
        - agent_flags
        - signals
        - map
        - agent_ids: list of agent ids. Use index if not provided.
        - readonly_slider: allow dragging slider

        Example:
        ---
        ```
        >>> trajectory, trajectory_mask, agent_mask, agent_records, agent_flags, signals = dataset[0]
        >>> rasterizer = SceneRasterizer(trajectory, trajectory_mask, agent_records, agent_flags, signals, map, config)
        ```
        '''
        self.map = map
        self.config = config

        
        self.trajectory = trajectory
        self.trajectory_mask = trajectory_mask
        self.agents = agents
        self.agent_flags = agent_flags
        self.signals = signals
        self.agent_ids = agent_ids or [*range(agents.size(0))]

        self.num_frames = self.config.seq_len

        self.init_params()
        self.init_ui(readonly_slider)

    def __del__(self):
        plt.close(self.ax.figure)

    def init_params(self):
        self.colors = {
            AgentType.car: 'dodgerblue', 
            AgentType.bicycle: 'green', 
            AgentType.motorcycle: 'orange', 
            AgentType.bus: 'turquoise',
            AgentType.truck: 'yellow', 
            AgentType.tricycle: 'hotpink', 
            AgentType.pedestrian: 'red'
        }

        # Define axes for the widgets
        self.current_frame: int = 0
        self.ColorsMap = { 'ax_slider': 'lightgoldenrodyellow' }
        self.delta_time = 1 / DRONE_FRAMERATE * self.config.stride # time between frames

        self.changed_button = False
        
        self.entities: dict[str, dict[str, dict[str, matplotlib.patches.Patch]]] = {
            "agents": {}, 
            "light": {}
        }

    def init_ui(self, readonly_slider: bool):
        '''
        init ui
        '''
        self.ax = self.map.plot()
        self.title = self.ax.title

        self.ax_slider = self.ax.figure.add_axes([0.2, 0.035, 0.2, 0.04], facecolor=self.ColorsMap['ax_slider'])  # Slider
        self.ax_button_play = self.ax.figure.add_axes([0.58, 0.035, 0.06, 0.04])
        self.ax_button_stop = self.ax.figure.add_axes([0.65, 0.035, 0.06, 0.04])

        self.centroid_style = dict(fill=True, edgecolor="black", lw=0.15, alpha=1, radius=0.2, zorder=30)
        self.track_style = dict(linewidth=1, zorder=10)
        self.track_style_future = dict(color="linen", linewidth=1, alpha=0.7, zorder=10)

        # Define the callbacks for the widgets' actions
        self.frame_slider = FrameControlSlider(
            self.ax_slider, 
            'Frame', 
            0, 
            self.num_frames - 1,
            valinit=self.current_frame,
            valfmt='%s'
        )

        self.button_play = Button(self.ax_button_play, u'▶️')
        self.button_stop = Button(self.ax_button_stop, '||')

        if not readonly_slider:
            self.frame_slider.on_changed(self.on_slider_change)

        self.button_play.on_clicked(self.on_play_btn_click)
        self.button_stop.on_clicked(self.on_stop_btn_click)

        self.timer = self.ax.figure.canvas.new_timer(interval=1000 * self.delta_time)
        self.timer.add_callback(self.on_tick, self.ax)

        self.ax.set_autoscale_on(True)
        self.update_figure()

    def trigger_update(self):
        '''
        update
        '''
        self.update_figure()
        self.frame_slider.set_val(self.current_frame)
        self.ax.figure.canvas.draw_idle()

    def on_slider_change(self, value: int):
        '''
        slider change event handler
        '''
        if not self.changed_button:
            self.current_frame = value
            self.trigger_update()
            
        self.changed_button = False

    def on_tick(self, _ = None):
        '''
        timer tick event handler
        '''
        if self.current_frame + 1 < self.num_frames:
            self.current_frame += 1
            self.changed_button = True
            self.trigger_update()
        else:
            self.on_stop_btn_click()

    def on_play_btn_click(self, _ = None):
        '''
        play button click event handler
        '''
        self.timer.start()

    def on_stop_btn_click(self, _ = None):
        '''
        stop button click event handler
        '''
        self.timer.stop()

    def update_figure(self):
        '''
        update plot
        '''
        self.ax.title.set_text(
            "Frame(s) = \n{} / {} ({:.2f}/{:.2f})".format(self.current_frame, self.num_frames - 1,
                                                           self.current_frame * self.delta_time,
                                                           (self.num_frames - 1) * self.delta_time))
        
        self.update_agents()
        self.update_signals()

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
        trajectory_mask = self.trajectory_mask[:, self.current_frame]
        trajectory = self.trajectory[:, self.current_frame]
        
        agents = self.agents
        agent_flags = self.agent_flags
        agent_ids = self.agent_ids

        future_trajectory = self.trajectory[:, self.current_frame:, :2]
        future_trajectory_mask = self.trajectory_mask[:, self.current_frame:]

        return trajectory, trajectory_mask, agents, agent_flags, agent_ids, future_trajectory, future_trajectory_mask

    def update_agents(self):
        '''
        update agent positions
        '''
        (
            trajectory, 
            trajectory_mask, 
            agents, 
            agent_flags, 
            agent_ids, 
            future_trajectory, 
            future_trajectory_mask
        ) = self.get_trajectory_data()

        mask = trajectory_mask & (agent_flags[:, 0] != AgentType.pedestrian) # pedestrian dont have length, width attributes
        
        bboxes, triangles = calculate_rot_bboxes_and_triangle(
            trajectory[mask, :2],
            agents[mask, 0],
            agents[mask, 1],
            trajectory[mask, 6]
        )

        bboxes = bboxes.numpy()
        triangles = triangles.numpy()

        bbox_pointer = 0

        for i in range(trajectory.size(0)):
            agent_id = agent_ids[i]

            # remove inactive agents
            if not trajectory_mask[i].item():
                if agent_id in self.entities['agents']:
                    for entity in self.entities['agents'][agent_id].values():
                        entity.remove()

                    del self.entities['agents'][agent_id]

                continue

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
                continue

            current_future_trajectory = future_trajectory[i, future_trajectory_mask[i]].numpy()

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
        '''
        get signals data at current frame
        '''
        return self.signals[self.current_frame]

    def update_signals(self):
        '''
        update traffic signals
        '''
        current_signals = self.get_signals_data()

        if current_signals.size(-1) == 1:
            current_signals = decode_traffic_signals(current_signals[None, :])[0]

        point_colors = { TrafficSignalType.red: 'red', TrafficSignalType.green: 'green', TrafficSignalType.yellow: 'yellow' }

        light_colors = {
            'red': { 'on': "#FC343E", 'off': "#440E11" },
            'yellow': { 'on': "#F5E049", 'off': "#423D14" },
            'green': { 'on': "#3AB549", 'off': "#103114" }
        }

        traffic_pos = [(22, 38), (32, 30), (32, 5), (25, -3), (5, -3), (-5, 5), (-5, 28), (3, 38)]

        for key in range(len(traffic_pos)):
            if key not in self.entities["light"]:
                self.entities["light"][key] = {}
                pos = traffic_pos[key]
                
                self.ax.add_patch(
                    plt.Rectangle(xy=(pos[0] - 1, pos[1] - 2.25), width=2, height=4.5, color='black', zorder=15)
                )

                pos = {'red': (pos[0], pos[1] + 1.5), 'yellow': pos, 'green': (pos[0], pos[1] - 1.5) }

                for value in point_colors.values():
                    onoff = 'on' if point_colors[current_signals[key].item()] == value else 'off'

                    circle = plt.Circle(pos[value], radius=0.5, zorder=19, color=light_colors[value][onoff])

                    self.ax.add_patch(circle)
                    self.entities["light"][key][value] = circle

            else:
                for value in point_colors.values():
                    onoff = 'on' if point_colors[current_signals[key].item()] == value else 'off'
                    self.entities["light"][key][value].set_color(light_colors[value][onoff])


class FrameControlSlider(Slider):
    def __init__(self, *args, **kwargs):
        self.inc = kwargs.pop('increment', 1)
        self.valfmt = '%s'
        Slider.__init__(self, *args, **kwargs)

    def set_val(self, val):
        if self.val != val:
            discrete_val = int(int(val / self.inc) * self.inc)
            super().set_val(discrete_val)