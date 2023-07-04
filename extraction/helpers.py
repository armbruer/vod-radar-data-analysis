from enum import Enum
from typing import List, Optional, Tuple
import numpy as np
import numba

from vod.frame.data_loader import FrameDataLoader
from vod.frame.labels import FrameLabels
from vod.frame.transformations import FrameTransformMatrix, homogenous_transformation_cartesian_coordinates
from vod.visualization.helpers import get_placed_3d_label_corners
        

"""
The DataVariant enum provides names for the fundamentally different approaches 
to work with the data in this dataset. 
In contrast to DataViews, DataVariants are cached for efficiency.
"""
class DataVariant(Enum):
    """
    The syntactic data contains the unannotated data only.
    """
    SYNTACTIC_DATA = 0,
    """
    This is also syntactic data but into two clases, 
    moving (dynamic) and not moving (static) detections.
    """
    SYNTACTIC_DATA_BY_MOVING = 1,
    """
    The semantic data is a summary of the annotated data.
    Each entry in this variant refers to a single detected object.
    """
    SEMANTIC_DATA = 2,
    """
    The semantic data  is a summary of the annotated data.
    Each entry in this variant refers to a single detected object.
    The data is split into several collections according to the classes.
    """
    SEMANTIC_DATA_BY_CLASS = 3
    
    """
    Returns a shortname for each datavariant that is useful e.g. for identifying files
    """
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
        """
        Returns a list of all data variants
        """
        return [DataVariant.SYNTACTIC_DATA, DataVariant.SYNTACTIC_DATA_BY_MOVING, 
                DataVariant.SEMANTIC_DATA, DataVariant.SEMANTIC_DATA_BY_CLASS]
    
    @staticmethod
    def basic_variants():
        """
        Returns a list of data variants that are basic in that they do not split their data.
        """
        return [DataVariant.SYNTACTIC_DATA, DataVariant.SEMANTIC_DATA]
    
    @staticmethod
    def split_variants():
        """
        Returns a list of data variants that split their data according to a criterion
        """
        return [DataVariant.SYNTACTIC_DATA_BY_MOVING, DataVariant.SEMANTIC_DATA_BY_CLASS]
    
    @staticmethod
    def syntactic_variants():
        """
        Returns a list of syntactic data variants
        """
        return [DataVariant.SYNTACTIC_DATA, DataVariant.SYNTACTIC_DATA_BY_MOVING]
    
    @staticmethod
    def semantic_variants():
        """
        Returns a list of semantic data variants
        """
        return [DataVariant.SEMANTIC_DATA, DataVariant.SEMANTIC_DATA_BY_CLASS]

    def column_names(self) -> List[str]:
        """
        Returns the column names of the current data variant
        """
        if self in DataVariant.syntactic_variants():
            return ["Frame Number", "Range [m]", 
                    "Azimuth [degree]", "Doppler [m/s]", "Elevation [degree]", 
                    "Doppler Compensated [m/s]", "x", "y", "z"]
        elif self in DataVariant.semantic_variants():
            # the Data Class stems directly from the dataset with no modification
            # the Class is summarized list of classes (that we are more interested in), see convert_to_summarized_class_id() below
            return ["Frame Number", "Range [m]", "Azimuth [degree]", "Doppler [m/s]", 
                    "Elevation [degree]", "Data Class", "Class", "Detections [#]", 
                    "Bbox volume [m^3]", "Doppler Compensated [m/s]", "x", "y", "z"]

        return []
    
    def subvariant_names(self) -> List[str]:
        """
        For those data variants that split their data according to a criterion, 
        this function returns a list of names for the subvariants.
        
        Returns a list of subvariant names
        """
        
        if self == DataVariant.SEMANTIC_DATA_BY_CLASS:
            return get_class_names()
        elif self == DataVariant.SYNTACTIC_DATA_BY_MOVING:
            return ["static", "dynamic"]
        
        return []

    def index_to_str(self, index) -> str:
        """
        For those data that split their data according to a criterion, 
        this function returns the subvariant name corresponding to the index
        
        :param index: the index of the subvariant name
        
        Returns the subvariant name corresponding to the given index
        """
        
        if self == DataVariant.SEMANTIC_DATA_BY_CLASS:
            return get_name_from_class_id(index, summarized=True)
        elif self == DataVariant.SYNTACTIC_DATA_BY_MOVING:
            return "static_rad" if index == 0 else "dynamic_rad"

        return ''
    
    def ticklabels(self) -> List[List[int]]:
        # must be in the order of the column_names(), see above
        
        to_str = lambda ticklabels: map(lambda l: map(str, l) if l is not None else None, ticklabels)
        
        if self in DataVariant.syntactic_variants():
            r = [0, 20, 40, 60, 80, 100]
            a = [-180, -135, -90, -45, 0, 45, 90, 135, 180]
            d = [-20, -10, 0, 10, 20]
            e = [-10, 0, 10, 30, 50]
            
            return to_str( [None, r, a, d, e, None, None, None, None])
        
        if self in DataVariant.semantic_variants():
            r = [0, 20, 40]
            a = [-90, -45, 0, 45, 90]
            d = [-20, -10, 0, 10, 20]
            e = [-10, 0, 10, 30, 50]
            
            return to_str([None, r, a, d, e, None, None, None, None, None, None, None, None])
        
    
    def lims(self) -> List[Tuple[int, int]]:
        # must be in the order of column_names(), see above
        if self in DataVariant.syntactic_variants():
            return [None, (0, 105), (-180, 180), (-30, 30), (-10, 90), None, None, None, None]
        elif self in DataVariant.semantic_variants():
            return [None, (0, 55), (-90, 90), (-30, 30), (-10, 50), None, None, None, None, None, None, None, None]

        return []

"""
The DataViewType class is intended to be used on top of a DataVariant.
It provides a view on top of the columns of the current variant by reducing 
the columns of a data variant to a subset.
"""
class DataViewType(Enum):
    """
    Keeps only range, azimuth, doppler columns.
    """
    RAD = 0,
    """
    Keeps only range, azimuth, doppler, elevation columns.
    """
    RADE = 1,
    """
    Keeps only columns that are required or useful for calculating stats about.
    """
    STATS = 2,
    """
    Keeps only the columns that are required 
    for creating longitute latitude plots showing the number of detections.
    """
    PLOT_LONG_LAT = 3,
    """
    Keeps only columns that make sense to be plotted by the plot_data() method.
    """
    EASY_PLOTABLE = 4,
    """
    Keeps only columns that are required or useful for a basic data analysis (RADE).
    """
    BASIC_ANALYSIS = 5,
    """
    Like BASIC_ANALYSIS, but including additional columns for semantic data.
    """
    EXTENDED_ANALYSIS = 6,
    """
    Keeps only x,y,z for plotting. This can be useful for debugging the other code.
    """
    PLOT_XYZ_ONLY = 7,
    """
    Keeps all columns, (no change to datavariant only)
    """
    NONE = 8
    
    def columns_to_drop(self) -> List[str]:
        """
        Returns a list of columns to drop for the current data view type.
        """
        if self == self.RAD:
            return ["Frame Number", "Data Class", "Class", "Doppler Compensated [m/s]", "Detections [#]", 
                    "Bbox volume [m^3]", "Elevation [degree]", "x", "y", "z"]
            
        if self == self.RADE:
            return ["Frame Number", "Data Class", "Class", "Doppler Compensated [m/s]", "Detections [#]", 
                    "Bbox volume [m^3]", "x", "y", "z"]
                
        elif self == self.STATS:
            return ["Frame Number", "Data Class", "Class", "x", "y", "z"]
                
        elif self == self.PLOT_LONG_LAT:
            return ["Frame Number", "Class", "Data Class", "Doppler Compensated [m/s]", 
                    "Bbox volume [m^3]", "Range [m]", "Azimuth [degree]", "Doppler [m/s]", "Elevation [degree]", "z"]
        
        elif self == self.EASY_PLOTABLE:
            return ["Frame Number", "Data Class", "Class", "x", "y", "z"]
        
        elif self == self.BASIC_ANALYSIS:
            return ["Data Class", "Class", "Doppler Compensated [m/s]", "Detections [#]", 
                    "Bbox volume [m^3]"]
        
        elif self == self.EXTENDED_ANALYSIS:
            return ["Data Class", "Class"]
        
        elif self == self.PLOT_XYZ_ONLY:
            return ["Frame Number", "Data Class", "Class", "Doppler Compensated [m/s]", "Detections [#]", 
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
    
    # Notes:
    # 1) See the radians formula https://en.wikipedia.org/wiki/Azimuth#In_cartography (we have origin (0, 0))
    # 2) See docs/figures/Prius_sensor_setup_5.png (radar) for the directions of x, y
    #    x = Longitudinal direction, y = Latitudinal direction
    # 3) We have mirrored the y values alongside the x axis (-y) in extract.py, so angles left of north (north=x-axis) will be negatives
    # 4) Additional note: https://stackoverflow.com/questions/283406/what-is-the-difference-between-atan-and-atan2-in-c
    #    conslusion: essentially always use arctan2 over arctan
    #    it is more stable, due to no y/x division, full 360 degrees output possible
    # 5) x and y are swapped in the numpy formula
    #    https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html
    # 
    x, y = list(locations.T)
    return np.arctan2(y, x) * 180 / np.pi

def elevation_angle_from_location(locations: np.ndarray) -> np.ndarray:
    """
    Returns the elevation angle in degrees for each location with respect to the origin (0, 0)
    Make sure the location coordinates are transformed to the radar coordinate system.
    
    :param locations: the locations array of shape (-1, 2)
    
    Returns the elevation angle in degrees to origin
    """
    
    x, y = list(locations.T)
    return np.arctan2(y, x) * 180 / np.pi

@numba.njit
def points_in_bbox(radar_points_radar: np.ndarray, radar_points_camera: np.ndarray, bbox: np.ndarray) -> Optional[List[np.ndarray]]:
    """
    Returns the radar points inside the given bounding box.
    Requires that radar points and bounding boxes are in the same coordinate system.
    The required order of the bounding box coordinates is shown below.

    :param radar_points: the radar points in cartesian and in the radar coordinate system
    :param radar_points: the radar points in cartesian and in the camera coordinate system
    :param bbox: the bounding box in cartesian and in the camera coordinate system (default)

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
    
    inside_points: List[np.ndarray] = []
    
    for i in range(radar_points_radar.shape[0]):
        x, y, z = radar_points_camera[i, :3]

        # the bounding box shape can be seen in transformed_3d_labels!
        # first index see order of corners above        
        # second index is x, y, z of the corner
        if x >= bbox[2, 0] and x <= bbox[1, 0] and y >= bbox[1, 1] and y <= bbox[0, 1] and z >= bbox[0, 2] and z <= bbox[4, 2]:
            # VERY IMPORTANT: return radar_points in the original radar coordinate system
            # otherwise azimuth and elevation calculation will be inherently flawed (due to the wrong origin!!!)
            inside_points.append(radar_points_radar[i])
            
    return None if not inside_points else inside_points
    
def get_data_for_objects_in_frame(loader: FrameDataLoader, transforms: FrameTransformMatrix) -> Optional[List[np.ndarray]]:
    """
    For each object in the frame retrieve the following data: frame number, ranges, azimuths, object class, summarized class, relative velocity compensated, 
    number of detections, bounding box volume, relative velocity (doppler), x, y, z.

    :param loader: the loader of the current frame
    :param transforms: the transformation matrix of the current frame

    Returns: a numpy array with the following columns: frame number, ranges, azimuths, object class, summarized class, relative velocity compensated, 
    number of detections, bounding box volume, relative velocity (doppler), x, y, z
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
    
    # Step Transform points into the same coordinate system as the labels
    radar_points_tr = homogenous_transformation_cartesian_coordinates(radar_data[:, :3], transform=transforms.t_camera_radar)
    radar_points_tr = np.hstack((radar_points_tr, radar_data[:, 3:]))
    
    frame_numbers: List[np.ndarray] = []
    #object_ids: List[np.ndarray] = [] # TODO future work
    object_clazz: List[np.ndarray] = [] # this class stems from the dataset
    summarized_clazz: List[np.ndarray] = [] # we summarize multiple classes here for easier plotting
    dopplers_compensated: List[np.ndarray] = [] # avg doppler, but compensated for ego-vehicle movement, per object
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
        points_matching = points_in_bbox(radar_points_radar=radar_data, radar_points_camera=radar_points_tr, bbox=bbox)
        
        if points_matching is not None:
            points_matching = np.vstack(points_matching)
            clazz_id = get_class_id_from_name(label['label_class'], summarized=False)
            summarized_id = convert_to_summarized_class_id(clazz_id)
            
            # Step 4: Get the avg doppler value of the object and collect it
            frame_numbers.append(loader.frame_number)
            object_clazz.append(clazz_id)
            summarized_clazz.append(summarized_id)
            dopplers_compensated.append(np.mean(points_matching[:, 5]))
            detections.append(points_matching.shape[0])
            bbox_vols.append(label['l'] * label['h'] * label['w'])            
             
            loc = np.array([[label['x'], label['y'], label['z']]])
            
            # transform from camera coordinates to radar coordinates, stay cartesian
            loc_radar = homogenous_transformation_cartesian_coordinates(loc, transforms.t_radar_camera)
            range_from_loc = locs_to_distance(loc_radar)            
            ranges.append(range_from_loc)
            
            # DO NOT USE radar_coordinates to calculate the azimuth and elevation
            # The problem: camera located behind radar
            # look at Prius_sensor_setup_5 camera coordinate system to understand indexes in the next lines
            # x is already mirrored, no reason to mirror it
            azimuths.append(azimuth_angle_from_location(loc[:, [2, 0]]))
            dopplers.append(np.mean(points_matching[:, 4]))
            elevations.append(elevation_angle_from_location(loc[:, [2, 1]]))
            
            # we need to use radar coordinates here to stay in one coordinate system between different data variants
            x.append(loc_radar[0, 0])
            y.append(loc_radar[0, 1])
            z.append(loc_radar[0, 2])
    
    if not object_clazz:
        return None
    
    columns = [frame_numbers, ranges, azimuths, dopplers, elevations, object_clazz, summarized_clazz, 
               detections, bbox_vols, dopplers_compensated, x, y, z]
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
        'cyclist', # includes both the bycicle and the rider
        'rider', # the human on top of the bycicle, motor, etc. separately
        'bicycle (unused)', # a bycicle that is currently not used, i.e. standing around
        'bicycle rack', # dt. ein Fahrradständer
        'human depiction', # this can be anything that looks remotely human, e.g. statue
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
