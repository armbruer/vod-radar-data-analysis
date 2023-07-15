from enum import Enum
from typing import List, Optional, Tuple
import numpy as np
from vod.frame.data_loader import FrameDataLoader
from vod.frame.labels import FrameLabels

from vod.frame.transformations import FrameTransformMatrix, homogenous_transformation_cart
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
            # the Class is summarized list of classes (that we are more interested in), 
            # see convert_to_summarized_class_id() below
            return ["Frame Number", "Range [m]", "Azimuth [degree]", "Doppler [m/s]", 
                    "Elevation [degree]", "Data Class", "Class", "Detections [#]", 
                    "Bbox volume [m^3]", "Doppler Compensated [m/s]", 
                    "Height [m]", "Width [m]", "Length [m]", "x", "y", "z"]

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
            return "static" if index == 0 else "dynamic"

        return ''
    
    def subvariant_name_to_index(self, subvariant_name) -> str:
        if self == DataVariant.SEMANTIC_DATA_BY_CLASS:
            return get_class_id_from_name(subvariant_name, summarized=True)
        elif self == DataVariant.SYNTACTIC_DATA_BY_MOVING:
            return 0 if subvariant_name == "static" else 1

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
            
            return to_str([None, r, a, d, e, None, None, None, None, 
                           None, None, None, None, None, None, None])
        
    
    def lims(self) -> List[Tuple[int, int]]:
        # must be in the order of column_names(), see above
        if self in DataVariant.syntactic_variants():
            return [None, (0, 105), (-180, 180), (-30, 30), (-10, 90), None, None, None, None]
        elif self in DataVariant.semantic_variants():
            return [None, (0, 55), (-90, 90), (-30, 30), (-10, 50), None, 
                    None, None, None, None, None, None, None, None, None, None]

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
    Keeps columns useful for a correlation heatmap.
    """
    CORR_HEATMAP = 5,
    """
    Keeps only columns that are useful when computing min, max
    """
    MIN_MAX_USEFUL = 7,
    """
    Keeps only x,y,z for plotting. This can be useful for debugging the other code.
    """
    PLOT_XYZ_ONLY = 8,
    """
    Keeps all columns, (no change to datavariant only)
    """
    NONE = 9
    
    def columns_to_drop(self) -> List[str]:
        """
        Returns a list of columns to drop for the current data view type.
        """
        if self == self.RAD:
            return ["Frame Number", "Data Class", "Class", "Doppler Compensated [m/s]", "Detections [#]", 
                    "Bbox volume [m^3]", "Elevation [degree]", "Height [m]", "Width [m]", "Length [m]", "x", "y", "z"]
            
        if self == self.RADE:
            return ["Frame Number", "Data Class", "Class", "Doppler Compensated [m/s]", "Detections [#]", 
                    "Bbox volume [m^3]", "Height [m]", "Width [m]", "Length [m]", "x", "y", "z"]
                
        elif self == self.STATS:
            return ["Frame Number", "Data Class", "Class", "x", "y", "z"]
                
        elif self == self.PLOT_LONG_LAT:
            return ["Frame Number", "Class", "Data Class", "Doppler Compensated [m/s]", 
                    "Height [m]", "Width [m]", "Length [m]", "Bbox volume [m^3]", "Range [m]", 
                    "Azimuth [degree]", "Doppler [m/s]", "Elevation [degree]", "z"]
        
        elif self == self.EASY_PLOTABLE:
            return ["Frame Number", "Data Class", "Class", "Height [m]", "Width [m]", "Length [m]", "x", "y", "z"]
        
        elif self == self.MIN_MAX_USEFUL:
            return ["Frame Number", "Data Class", "Class", "x", "y", "z"]
        
        elif self == self.PLOT_XYZ_ONLY:
            return ["Frame Number", "Data Class", "Class", "Doppler Compensated [m/s]", "Detections [#]", 
                    "Bbox volume [m^3]", "Range [m]", "Azimuth [degree]", "Doppler [m/s]", "Elevation [degree]",
                    "Height [m]", "Width [m]", "Length [m]"]
            
        elif self == self.CORR_HEATMAP:
            return ["Frame Number", "Data Class", "Class", "x", "y", "z"]

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
    return np.arctan2(-y, x) * 180 / np.pi  # flip y for correct angles (see prius image)

def elevation_angle_from_location(locations: np.ndarray) -> np.ndarray:
    """
    Returns the elevation angle in degrees for each location with respect to the origin (0, 0)
    Make sure the location coordinates are transformed to the radar coordinate system.
    
    :param locations: the locations array of shape (-1, 2)
    
    Returns the elevation angle in degrees to origin
    """
    
    x, y = list(locations.T)
    return np.arctan2(y, x) * 180 / np.pi


def get_bbox_transformation_matrix(bbox_placed: np.ndarray):
    # order of corners of bbox
    #    5--------4 
    #   /|       /| | height (z)
    #  / |      / | |
    # 6--------7  | |
    # |  |     |  |
    # |  1-----|--0 ^ length (x)
    # | /      | / /
    # |/       |/ /
    # 2--------3 ---> width (y)
    #
    
    # bbox coordinate system (as above)
    # ^z ^x
    # | /
    # |/
    # --->y
    
    # camera and radar coordinate system
    # this is the coordinate system of the bbox_placed parameter
    # (see prius file)
    # ^z ^y
    # | /
    # |/
    # --->x 
    
    # we don't know after the transformation whether our axis all point into the positive direction
    # meaning e.g. point 3 could have a negative y coordinate
    # so we need to keep track of this additionally
    
    # we would later simply like to know whether the points are within bbox_limits
    # for this we need positive unit vectors, so we need to take the appropriate origin
    # but we do not know which one this is, so we need to check
    
    # origin_index = -1
    # for i in bbox_placed.shape[0]:
    #     if ((i <= 3 and 
    #         bbox_placed[((i + 1) % 4)] - bbox_placed[i] >= 0 and 
    #         bbox_placed[((i - 1) % 4)] - bbox_placed[i] >= 0 and 
    #         bbox_placed[(i + 4)] - bbox_placed[i] >= 0) or 
    #         (i > 3 and 
    #         bbox_placed[((i + 1) % 8)] - bbox_placed[i] >= 0 and 
    #         bbox_placed[((i - 1) % 8)] - bbox_placed[i] >= 0 and 
    #         bbox_placed[(i - 4)] - bbox_placed[i] >= 0
    #          )):
    #         origin_index = i
    #         break
            
    # assert origin_index != -1
    # neighbor_indexes = (origin_index + 1) % 4, (origin_index - 1) % 4, origin_index + 4 if origin_index <= 3 else (origin_index + 1) % 8, (origin_index - 1) % 8, origin_index - 4 if origin_index <= 3
        
    
    x_vec = bbox_placed[1] - bbox_placed[2]
    y_vec = bbox_placed[3] - bbox_placed[2]
    z_vec = bbox_placed[6] - bbox_placed[2]
    
    x_vec = list(x_vec / np.linalg.norm(x_vec))
    y_vec = list(y_vec / np.linalg.norm(y_vec))
    z_vec = list(z_vec / np.linalg.norm(z_vec))
    
    rotation_matrix = np.linalg.inv(np.array([x_vec, y_vec, z_vec]))
    rotation_matrix_hom = np.hstack((
        rotation_matrix, np.zeros((rotation_matrix.shape[1], 1), dtype=np.float32)))
    
    # translation from old origin to new origin (old is (0, 0, 0))
    translation_hom = [list(-bbox_placed[2]) + [1]]
    
    translation_matrix = np.vstack((rotation_matrix_hom, translation_hom)).T
    
    return translation_matrix
    
def points_in_bbox(radar_points: np.ndarray, 
                   bbox_limits: np.ndarray,
                   transformation_matrix: np.ndarray = np.identity(4)) -> Optional[np.ndarray]:
    """
    Returns the radar points inside the given bounding box.
    Requires that radar points and bounding boxes are in the same coordinate system.
    The required order of the bounding box coordinates is shown below.

    :param radar_points: the radar points in cartesian and in the radar coordinate system
    :param bbox_limits: height, width and length of the bbox

    Returns: radar points inside the given bounding box
    """
    
    # order of corners
    #    5--------4 
    #   /|       /| | height (z)
    #  / |      / | |
    # 6--------7  | |
    # |  |     |  |
    # |  1-----|--0 ^ length (x)
    # | /      | / /
    # |/       |/ /
    # 2--------3 ---> width (y)
    
    inside_points: List[np.ndarray] = []
    
    # transform radar coordinates to local reference frame of the cube with origin at point 2
    radar_points_tr = homogenous_transformation_cart(radar_points[:, :3], transformation_matrix)
    radar_points_tr = np.hstack((radar_points_tr, radar_points[:, 3:]))
    
    # the bounding box has shape (8, 3)
    # first index see order of corners above        
    # second index is x, y, z of the corner
    
    inside_points: np.ndarray = radar_points[np.where(
        (radar_points_tr[:, 0] <= bbox_limits[0]) &
        (radar_points_tr[:, 1] <= bbox_limits[1]) &
        (radar_points_tr[:, 2] <= bbox_limits[2]) &
        
        (radar_points_tr[:, 0] >= 0) &
        (radar_points_tr[:, 1] >= 0) &
        (radar_points_tr[:, 2] >= 0))]
            
    return inside_points if inside_points.size != 0 else None


def prepare_radar_data(loader: FrameDataLoader):
    # radar_points shape: [x, y, z, RCS, v_r, v_r_compensated, time] (-1, 7)
    radar_data_r = loader.radar_data
    if radar_data_r is None:
        return None
    
    # we don't want to include points from previous scans, i.e. accumulated points
    # not needed probably, just to be safe
    radar_data_r = radar_data_r[radar_data_r[:, 6] == 0]
    return radar_data_r


def find_matching_points_for_bboxes(radar_points: np.ndarray,
                                    labels: FrameLabels,
                                    transforms: FrameTransformMatrix,
                                    camera_coordinates: bool = False) -> List[Tuple[dict, Optional[np.ndarray]]]:

    labels_3d_corners = get_placed_3d_label_corners(labels=labels, transforms=transforms, camera_coordinates=camera_coordinates)
    
    matching_points: List[np.ndarray] = []
    for label in labels_3d_corners:
        bbox_placed = label['corners_3d_placed']
        print(label['label_class'])
        
        # we need to apply a rotation matrix as the bbox is not aligned with the axis of the coordinate system
        transformation_matrix: np.ndarray = get_bbox_transformation_matrix(bbox_placed)
        bbox_limits = np.array([label['l'], label['w'], label['h']])
        matching_points_r = points_in_bbox(radar_points=radar_points, 
                                                bbox_limits=bbox_limits, transformation_matrix=transformation_matrix)
        
        matching_points.append((label, matching_points_r))
        
    return matching_points

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
    return 9 if class_id >= 9 else class_id
    
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
