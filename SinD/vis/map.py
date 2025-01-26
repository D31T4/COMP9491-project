import numpy as np

import matplotlib.pyplot as plt

import xml.etree.ElementTree as xml

from SinD.config import get_map_path
from SinD.utils.geom import alpha_shape

from typing import Optional

# average distance between 1 deg of latitude
DEG_DIST = 111319.44

def latlon2xy(latlon: np.ndarray) -> np.ndarray:
    '''
    Project to tangent plane centered at (0, 0). Since Earth is big compared to the intersection, error is negligible.
    '''
    assert len(latlon.shape) == 2
    assert latlon.shape[1] == 2

    cart = np.zeros_like(latlon)

    lat0 = 0
    long0 = 0

    dist_lat = DEG_DIST
    dist_long = DEG_DIST * np.cos(np.radians(lat0))

    cart[:, 0] = dist_long * (latlon[:, 1] - long0)
    cart[:, 1] = dist_lat * (latlon[:, 0] - lat0)

    return cart

def get_type(element: xml.Element):
    for tag in element.findall("tag"):
        if tag.get("k") == "type":
            return tag.get("v")
    
    return None


def get_subtype(element: xml.Element):
    for tag in element.findall("tag"):
        if tag.get("k") == "subtype":
            return tag.get("v")
        
    return None

def build_polyline(element: xml.Element, point_dict: dict[int, np.ndarray]) -> np.ndarray:
    xy = []

    for nd in element.findall("nd"):
        pt_id = int(nd.get("ref"))
        xy.append(point_dict[pt_id][None, :])

    return np.concatenate(xy)
    
class IntersectionMap:
    def __init__(self, path: Optional[str] = None):
        if path is None:
            path = get_map_path()

        xml_root = xml.parse(path).getroot()

        point_dict = dict()

        for node in xml_root.findall("node"):
            xy = latlon2xy(np.array([[
                float(node.get('lat')), 
                float(node.get('lon'))
            ]], dtype=np.float32))

            point_dict[int(node.get('id'))] = xy[0]

        self.min_x = float('inf')
        self.min_y = float('inf')
        self.max_x = -float('inf')
        self.max_y = -float('inf')

        for point in point_dict.values():
            self.min_x = min(point[0], self.min_x)
            self.min_y = min(point[1], self.min_y)
            self.max_x = max(point[0], self.max_x)
            self.max_y = max(point[1], self.max_y)

        self.polylines: list[tuple[str, np.ndarray]] = []
        self.lanelets: list[np.ndarray] = []

        polyline_dict: dict[int, np.ndarray] = dict()

        for way in xml_root.findall('way'):
            way_type = get_type(way)

            if way_type not in ['curbstone', 'line_thin', 'line_thick', 'pedestrian_marking', 'bike_marking', 'stop_line', 'virtual', 'road_border', 'guard_rail', 'wait_line', 'zebra_marking']:
                continue

            if way_type in ['line_thin', 'line_thick']:
                way_subtype = get_subtype(way)

                if way_subtype == 'dashed':
                    way_type = f'{way_type}:dashed'

            polyline = build_polyline(way, point_dict)

            self.polylines.append((way_type, polyline))

            polyline_dict[int(way.get('id'))] = polyline

        lanepoints: list[np.ndarray] = []

        for relation in xml_root.findall('relation'):
            relation_type = get_type(relation)

            if relation_type != 'lanelet':
                continue

            left, right = relation.findall('member')
            
            left_id = int(left.attrib['ref'])
            right_id = int(right.attrib['ref'])

            left_points = polyline_dict[left_id]
            right_points = polyline_dict[right_id]

            lanepoints.append(np.concatenate([left_points, right_points]))


        # 5 is a magic number
        self.lanelets = alpha_shape(np.concatenate(lanepoints), 5)
        

    def plot(self):
        plt.figure()
        ax = plt.subplot()

        ax.set_xlim([self.min_x - 10, self.max_x + 10])
        ax.set_ylim([self.min_y - 10, self.max_y + 10])
        ax.axis('off')

        ax.set_aspect('equal', adjustable='box')

        for line_type, polyline in self.polylines:
            match line_type:
                case "curbstone":
                    format_dict = dict(color="black", linewidth=1, zorder=11)
                case "line_thin":
                    format_dict = dict(color="white", linewidth=1, zorder=10)
                case 'line_thin:dashed':
                    format_dict = dict(color="white", linewidth=1, zorder=10)
                case "line_thick":
                    format_dict = dict(color="white", linewidth=2, zorder=10)
                case 'line_thick:dashed':
                    format_dict = dict(color="white", linewidth=2, zorder=10, dashes=[10, 10])
                case "pedestrian_marking":
                    format_dict = dict(color="white", linewidth=1, zorder=10, dashes=[5, 10])
                case "bike_marking":
                    format_dict = dict(color="white", linewidth=1, zorder=10, dashes=[5, 10])
                case "stop_line":
                    format_dict = dict(color="white", linewidth=3, zorder=10)
                case "virtual":
                    format_dict = dict(color="blue", linewidth=1, zorder=10, dashes=[2, 5])
                    continue
                case "road_border":
                    format_dict = dict(color="orange", linewidth=1, zorder=10)
                case "guard_rail":
                    format_dict = dict(color="black", linewidth=1, zorder=10)
                case "wait_line":
                    format_dict = dict(color="yellow", linewidth=2, zorder=10, dashes = [2, 3])
                case "zebra_marking":
                    format_dict = dict(color="white", zorder=10, linewidth=2)
                case _:
                    continue

            ax.plot(polyline[:, 0], polyline[:, 1], **format_dict)
        
        for points in self.lanelets:
            polygon = plt.Polygon(points, closed=True, color='lightgray')
            ax.add_patch(polygon)

        return ax
