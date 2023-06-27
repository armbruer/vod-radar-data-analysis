from enum import Enum
from typing import List, Optional
import numpy as np

from vod.frame.data_loader import FrameDataLoader
from vod.frame.labels import FrameLabels
from vod.frame.transformations import FrameTransformMatrix, homogenous_transformation_cartesian_coordinates
from vod.visualization.helpers import get_placed_3d_label_corners
        

class DataVariant(Enum):
    SYNTACTIC_DATA = 0,
    SYNTACTIC_DATA_BY_MOVING = 1,
    SEMANTIC_DATA = 2,
    SEMANTIC_DATA_BY_CLASS = 3
    
    def shortname(self):
        if self == self.SYNTACTIC_DATA:
            return 'syn'
        elif self == self.SEMANTIC_DATA:
            return 'sem'
        elif self == self.SYNTACTIC_DATA_BY_MOVING:
            return 'syn_by_moving'
        elif self == self.SEMANTIC_DATA_BY_CLASS:
            return 'sem_by_class'
    
    @staticmethod
    def all_variants():
        return [DataVariant.SYNTACTIC_DATA, DataVariant.SYNTACTIC_DATA_BY_MOVING, 
                DataVariant.SEMANTIC_DATA, DataVariant.SEMANTIC_DATA_BY_CLASS]
    
    @staticmethod
    def basic_variants():
        return [DataVariant.SYNTACTIC_DATA, DataVariant.SEMANTIC_DATA]
    
    @staticmethod
    def split_variants():
        return [DataVariant.SYNTACTIC_DATA_BY_MOVING, DataVariant.SEMANTIC_DATA_BY_CLASS]
    
    @staticmethod
    def syntactic_variants():
        return [DataVariant.SYNTACTIC_DATA, DataVariant.SYNTACTIC_DATA_BY_MOVING]
    
    @staticmethod
    def semantic_variants():
        return [DataVariant.SEMANTIC_DATA, DataVariant.SEMANTIC_DATA_BY_CLASS]

    def column_names(self) -> List[str]:
        if self in DataVariant.syntactic_variants():
            return ["Frame Number", "Range [m]", "Azimuth [degree]", "Doppler [m/s]", "Elevation [degree]", "x", "y", "z"]
        elif self in DataVariant.semantic_variants():
            # the Data Class stems directly from the dataset with no modification
            # the Class is summarized list of classes, see convert_to_summarized_class_id() below
            return ["Frame Number", "Data Class", "Class", "Velocity [m/s]", "Detections [#]", 
                    "Bbox volume [m^3]", "Range [m]", "Azimuth [degree]", "Doppler [m/s]", "Elevation [degree]", "x", "y", "z"]

        return []
    
    def subvariant_names(self) -> List[str]:
        if self == DataVariant.SEMANTIC_DATA_BY_CLASS:
            return get_class_names()
        elif self == DataVariant.SYNTACTIC_DATA_BY_MOVING:
            return ["static", "dynamic"]
        
        return []

    def index_to_str(self, index) -> str:
        if self == DataVariant.SEMANTIC_DATA_BY_CLASS:
            return get_name_from_class_id(index, summarized=True)
        elif self == DataVariant.SYNTACTIC_DATA_BY_MOVING:
            return "static_rad" if index == 0 else "dynamic_rad"

        return ''


class DataView(Enum):
    RAD = 0,
    RADE = 1,
    STATS = 2,
    PLOT_XY = 3,
    PLOTABLE = 4,
    PLOT_DETECTIONS_MAP = 5,
    BASIC_ANALYSIS = 6,
    EXTENDED_ANALYSIS = 7,
    PLOT_XYZ_ONLY = 8,
    NONE = 9
    
    def columns_to_drop(self) -> List[str]:
        if self == self.RAD:
            return ["Frame Number", "Data Class", "Class", "Velocity [m/s]", "Detections [#]", 
                    "Bbox volume [m^3]", "Elevation [degree]", "x", "y", "z"]
            
        if self == self.RADE:
            return ["Frame Number", "Data Class", "Class", "Velocity [m/s]", "Detections [#]", 
                    "Bbox volume [m^3]", "x", "y", "z"]
                
        elif self == self.STATS:
            return ["Frame Number", "Data Class", "Class", "x", "y", "z"]
                
        elif self == self.PLOT_XY:
            return ["Frame Number", "Class", "Data Class", "Velocity [m/s]", 
                    "Bbox volume [m^3]", "Range [m]", "Azimuth [degree]", "Doppler [m/s]", "Elevation [degree]", "z"]
        
        elif self == self.PLOTABLE:
            return ["Frame Number", "Data Class", "Class", "x", "y", "z"]
        
        elif self == self.BASIC_ANALYSIS:
            return ["Data Class", "Class", "Velocity [m/s]", "Detections [#]", 
                    "Bbox volume [m^3]", "x", "y", "z"]
        
        elif self == self.EXTENDED_ANALYSIS:
            return ["Data Class", "Class", "x", "y", "z"]
        
        elif self == self.PLOT_DETECTIONS_MAP:
            return ["Data Class", "Frame Number", "Detections [#]", "x", "y"]
        
        elif self == self.PLOT_XYZ_ONLY:
            return ["Frame Number", "Data Class", "Class", "Velocity [m/s]", "Detections [#]", 
                    "Bbox volume [m^3]", "Range [m]", "Azimuth [degree]", "Doppler [m/s]", "Elevation [degree]"]
        
        # NONE
        return []
            

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
    
    :param locations: the locations array of shape (-1, 2)
    
    Returns the azimuth angle in degrees to origin
    """
    
    # see the radians formula here (we have origin (0, 0))
    # https://en.wikipedia.org/wiki/Azimuth#In_cartography
    # IMPORTANT: see docs/figures/Prius_sensor_setup_5.png (radar) for the directions of x, y = row[0], row[1]
    # x = Longitudinal direction, y = Latitudinal direction
    # x<->y are swapped with the forumla from Wikipedia
    return np.rad2deg(np.apply_along_axis(lambda row: np.arctan2(row[1], row[0]), 1, locations))

def elevation_angle_from_location(locations: np.ndarray) -> np.ndarray:
    """
    Returns the elevation angle in degrees for each location with respect to the origin (0, 0)
    Make sure the location coordinates are transformed to the radar coordinate system.
    
    :param locations: the locations array of shape (-1, 2)
    
    Returns the elevation angle in degrees to origin
    """
    
    # we can use the same formula as above
    return np.rad2deg(np.apply_along_axis(lambda row: np.arctan2(row[1], row[0]), 1, locations))  

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
    
    frame_numbers: List[np.ndarray] = []
    object_ids: List[np.ndarray] = [] # TODO future work
    object_clazz: List[np.ndarray] = [] # this class stems from the dataset
    plot_clazz: List[np.ndarray] = [] # we summarize multiple classes here for easier plotting
    velocity_abs: List[np.ndarray] = [] # one avg absolute velocity per bounding box
    dopplers: List[np.ndarray] = [] # one avg doppler value per bounding box
    detections: List[np.ndarray] = [] # number of radar_points inside a bounding box
    bbox_vols: List[np.ndarray] = [] # bounding box volume in m^3
    ranges: List[np.ndarray] = [] # range in m
    azimuths: List[np.ndarray] = [] # azimuth in degrees
    elevations: List[np.ndarray] = [] # elevation in degrees
    
    # IMPORTANT: see docs/figures/Prius_sensor_setup_5.png (radar) for the directions of these variables
    x: List[np.ndarray] = []
    y: List[np.ndarray] = []
    z: List[np.ndarray] = []
    
    for label in labels_with_corners:
        # Step 3: For each bounding box get a list of radar points which are inside of it
        bbox = label['corners_3d_placed']
        points_matching = points_in_bbox(radar_points=radar_data_transformed, bbox=bbox)
        
        if points_matching is not None:
            clazz_id = get_class_id_from_name(label['label_class'], summarized=False)
            summarized_id = convert_to_summarized_class_id(clazz_id)
            
            # Step 4: Get the avg doppler value of the object and collect it
            frame_numbers.append(loader.frame_number)
            object_clazz.append(clazz_id)
            plot_clazz.append(summarized_id)
            velocity_abs.append(np.mean(points_matching[:, 5]))
            detections.append(points_matching.shape[0])
            bbox_vols.append(label['l'] * label['h'] * label['w'])            
            
            # flip the y-axis, so left of 0 is negative and right is positive as one would expect it in plots
            # see docs/figures/Prius_sensor_setup_5.png (radar) 
            loc = np.array([[label['x'], label['y'], label['z']]])
            # transform from camera coordinates to radar coordinates, stay cartesian
            loc_transformed = homogenous_transformation_cartesian_coordinates(loc, transforms.t_radar_camera)
            loc_transformed[0, 1] = -loc_transformed[0, 1]
            range_from_loc = locs_to_distance(loc_transformed)
            
            ranges.append(range_from_loc)
            azimuths.append(azimuth_angle_from_location(loc_transformed[:, :2]))
            dopplers.append(np.mean(points_matching[:, 4]))
            elevations.append(elevation_angle_from_location(loc_transformed[:, [0, 2]]))
            
            x.append(loc_transformed[0, 0])
            y.append(loc_transformed[0, 1])
            z.append(loc_transformed[0, 2])
    
    if not object_clazz:
        return None
    
    columns = [frame_numbers, object_clazz, plot_clazz, velocity_abs, 
               detections, bbox_vols, ranges, azimuths, dopplers, elevations, x, y, z]
    return list(map(np.hstack, columns))


def get_class_names(summarized: bool = True) -> List[str]:
    """
    Get a list of class names.
    
    :param summarize: whether to summarize classes for easier plotting

    Returns: a list of class names
    """
    
    if summarized:
        return [
        'car',
        'pedestrian',
        'cyclist',
        'rider',
        'bicycle (unused)',
        'bicycle rack', # dt. ein Fahrradständer
        'human depiction',
        'moped or scooter',
        'motor',
        'other']
    
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
    
def convert_to_summarized_class_id(class_id: int) -> int:
    # we summarize the last four classes to one
    return 9 if class_id >=9 else class_id
    
def get_class_ids(summarized: bool = True) -> List[int]:
    """
    Returns a list of class ids

    Returns: a list of class ids
    """
    return list(range( 10 if summarized else 13))

def get_name_from_class_id(clazz: int, summarized: bool = True) -> str:
    """
    Returns the name corresponding to the given class id.
    
    :param clazz: the class id
    :param summarized: 
    
    Returns: the name corresponding to the given class id
    """
    
    class_id_to_name = {i: n for i, n in enumerate(get_class_names(summarized))}
    return class_id_to_name[clazz]
    
    
def get_class_id_from_name(name: str, summarized: bool = True) -> int:
    """
    Returns the class id corresponding to the given class name.
    
    :param name: the name of the class
    :param summarized: 
    
    Returns: the class id corresponding to the given class name
    """
    
    name_to_class_id = {n: i for i, n in enumerate(get_class_names(summarized))}
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