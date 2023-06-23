from enum import Enum
from typing import List, Optional
import numpy as np

from vod.frame.data_loader import FrameDataLoader
from vod.frame.labels import FrameLabels
from vod.frame.transformations import FrameTransformMatrix, homogenous_transformation_cartesian_coordinates
from vod.visualization.helpers import get_placed_3d_label_corners


class DataVariant(Enum):
    SYNTACTIC_RAD = 0,
    SEMANTIC_RAD = 1,
    STATIC_DYNAMIC_RAD = 2,
    SEMANTIC_OBJECT_DATA = 3,
    SEMANTIC_OBJECT_DATA_BY_CLASS = 4

    def column_names(self) -> List[str]:
        if self == DataVariant.SEMANTIC_RAD or self == DataVariant.STATIC_DYNAMIC_RAD or self == DataVariant.SYNTACTIC_RAD:
            return ["frame number", "range (m)", "azimuth (degree)", "doppler (m/s)"]
        elif self == DataVariant.SEMANTIC_OBJECT_DATA_BY_CLASS or self == DataVariant.SEMANTIC_OBJECT_DATA:
            return ["frame number", "class", "velocity (m/s)", "detections (#)", "bbox volume (m^3)", "range (m)", "azimuth (degree)", "doppler (m/s)"]

        return []
    
    def subvariants(self) -> List[str]:
        if self == DataVariant.SEMANTIC_OBJECT_DATA_BY_CLASS:
            return get_class_names()
        elif self == DataVariant.STATIC_DYNAMIC_RAD:
            return ["static_rad", "dynamic_rad"]

    def index_to_str(self, index) -> str:
        if self == DataVariant.SEMANTIC_OBJECT_DATA_BY_CLASS:
            return get_name_from_class_id(index)
        elif self == DataVariant.STATIC_DYNAMIC_RAD:
            if index == 0:
                return "static_rad"
            else:
                return "dynamic_rad"

        return ''


def locs_to_distance(locations: np.ndarray) -> np.ndarray:
    """
    Return the distance to the origin (0, 0, 0) for a given location array of shape (-1, 3)
    Make sure the location coordinates are transformed to the radar coordinate system.
    
    :param locations: the locations array of shape (-1, 3)
    
    Returns the distance to origin
    """
    
    return np.apply_along_axis(lambda row: np.linalg.norm(row), 1, locations)

def azimuth_angle_from_location(locations: np.ndarray) -> np.ndarray:
    """
    Returns the azimuth angle in degrees for each location with respect to the origin (0, 0)
    Make sure the location coordinates are transformed to the radar coordinate system.
    
    :param locations: the locations array of shape (-1, 2)+
    
    Returns the azimuth angle in degrees to origin
    """
    
    # see the radians formula here (we have origin (0, 0))
    # https://en.wikipedia.org/wiki/Azimuth#In_cartography
    return np.rad2deg(np.apply_along_axis(lambda row: np.arctan2(row[0], row[1]), 1, locations))
   
def points_in_bbox(radar_points: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
    """
    Returns the radar points inside the given bounding box.
    Requires that radar points and bounding boxes are in the same coordinate system.
    The required order of the bounding box coordinates is shown below.

    :param radar_points: the radar points in cartesian
    :param bbox: the bounding box in cartesian

    Returns: radar points inside the given bounding box
    """
    
    # order of corners
    #    7--------4
    #   /|       /|
    #  / |      / |
    # 6--------5  |
    # |  |     |  |
    # |  3-----|--0
    # | /      | /
    # |/       |/
    # 2--------1
    
    inside_points = []
    
    for i in range(radar_points.shape[0]):
        radar_point = radar_points[i, :3]
        x, y, z = radar_point

        # the bounding box shape can be seen in transformed_3d_labels!
        # first index see order of corners above        
        # second index is x, y, z of the corner
        if x >= bbox[2, 0] and x <= bbox[1, 0] and y >= bbox[1, 1] and y <= bbox[0, 1] and z >= bbox[0, 2] and z <= bbox[4, 2]:
            inside_points.append(radar_points[i])
            
    if not inside_points:
        return None
            
    return np.vstack(inside_points)
    
def get_data_for_objects_in_frame(loader: FrameDataLoader, transforms: FrameTransformMatrix) -> Optional[List[np.ndarray]]:
    """
    For each object in the frame retrieve the following data: frame number, object class, absolute velocity, 
    number of detections, bounding box volume, ranges, azimuths, relative velocity (doppler).

    :param loader: the loader of the current frame
    :param transforms: the transformation matrix of the current frame

    Returns: a numpy array with the following columns: frame number, object class, absolute velocity, 
    number of detections, bounding box volume, ranges, azimuths, relative velocity (doppler)
    """
    
    labels = loader.get_labels()
    if labels is None:
        return None
    
    labels = FrameLabels(labels)
    
    # Step 1: Obtain corners of bounding boxes and radar data points
    labels_with_corners = get_placed_3d_label_corners(labels)
    
    # radar_points shape: [x, y, z, RCS, v_r, v_r_compensated, time] (-1, 7)
    radar_data = loader.radar_data
    if radar_data is None:
        return None
    
    # Step 2: Transform points and labels into same coordinate system
    radar_points = homogenous_transformation_cartesian_coordinates(radar_data[:, :3], transform=transforms.t_camera_radar)
    radar_data_transformed = np.hstack((radar_points, loader.radar_data[:, 3:]))
    
    
    frame_numbers = []
    object_ids = [] # TODO future work
    object_clazz = []
    velocity_abs = [] # one avg absolute velocity per bounding box
    dopplers = [] # one avg doppler value per bounding box
    detections = [] # number of radar_points inside a bounding box
    bbox_vols = [] # bounding box volume
    ranges = [] # range in m
    azimuths = [] # azimuth in degree
    
    
    for label in labels_with_corners:
        # Step 3: For each bounding box get a list of radar points which are inside of it
        bbox = label['corners_3d_placed']
        points_matching = points_in_bbox(radar_points=radar_data_transformed, bbox=bbox)
        
        if points_matching is not None:
            # Step 4: Get the avg doppler value of the object and collect it
            frame_numbers.append(loader.frame_number)
            object_clazz.append(get_class_id_from_name(label['label_class']))
            velocity_abs.append(np.mean(points_matching[:, 5]))
            detections.append(points_matching.shape[0])
            bbox_vols.append(label['l'] * label['h'] * label['w'])            
            
            loc = np.array([[label['x'], label['y'], label['z']]])
            loc_transformed = homogenous_transformation_cartesian_coordinates(loc, transforms.t_radar_camera)
            range_from_loc = locs_to_distance(loc_transformed)
            ranges.append(range_from_loc)
            
            azimuths.append(azimuth_angle_from_location(loc_transformed[:, :2]))
            dopplers.append(np.mean(points_matching[:, 4]))
    
    if not object_clazz:
        return None
    
    columns = [frame_numbers, object_clazz, velocity_abs, detections, bbox_vols, ranges, azimuths, dopplers]
    return list(map(np.hstack, columns))


def get_class_names() -> List[str]:
    """
    Get a list of class names.

    Returns: a list of class names
    """
    
    return [
        'Car',
        'Pedestrian',
        'Cyclist',
        'rider',
        'bicycle',
        'bicycle_rack',
        'human_depiction',
        'moped_scooter',
        'motor',
        'ride_other',
        'ride_uncertain',
        'truck',
        'vehicle_other']
    
def get_class_ids() -> List[int]:
    """
    Returns a list of class ids

    Returns: a list of class ids
    """
    return list(range( 13))

def get_name_from_class_id(clazz: int) -> str:
    """
    Returns the name corresponding to the given class id.
    
    :param clazz: the class id
    
    Returns: the name corresponding to the given class id
    """
    class_id_to_name = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist',
        3: 'rider',
        4: 'bicycle',
        5: 'bicycle_rack',
        6: 'human_depiction',
        7: 'moped_scooter',
        8: 'motor',
        9: 'ride_other',
        10: 'ride_uncertain',
        11: 'truck',
        12: 'vehicle_other'
    }
    
    return class_id_to_name[clazz]
    
    
def get_class_id_from_name(name: str) -> int:
    """
    Returns the class id corresponding to the given class name.
    
    :param name: the name of the class
    
    Returns: the class id corresponding to the given class name
    """
    
    name_to_class_id = {
       'Car': 0,
        'Pedestrian' : 1,
        'Cyclist' : 2,
        'rider' : 3,
        'bicycle' : 4,
        'bicycle_rack' : 5,
        'human_depiction' : 6,
        'moped_scooter' : 7,
        'motor' : 8,
        'ride_other' : 9,
        'ride_uncertain' : 10,
        'truck' : 11,
        'vehicle_other' : 12
    }
    
    return name_to_class_id[name]    


def vis_to_debug(loader: FrameDataLoader) -> None:
    """
    Helper function to visually debug a frame. 
    Provides visualization of the current frame.
    Best used from a jupyter notebook.
    
    :param loader: the loader of the frame to be debugged
    """
    
    from vod.visualization import Visualization3D
    vis3d = Visualization3D(loader)

    vis3d.draw_plot(radar_origin_plot = True,
                  lidar_origin_plot = True,
                  camera_origin_plot = True,
                  lidar_points_plot = True,
                  radar_points_plot = True,
                  annotations_plot = True)
    
    from vod.visualization import Visualization2D
    vis2d = Visualization2D(loader)
    
    vis2d.draw_plot(show_lidar=True,
                             show_radar=True,
                             show_gt=True,
                             min_distance_threshold=5,
                             max_distance_threshold=40,
                             save_figure=True)