import numpy as np
from typing import List

from vod.frame.data_loader import FrameDataLoader
from vod.frame.labels import FrameLabels
from vod.frame.transformations import FrameTransformMatrix, homogenous_transformation_cartesian_coordinates
from vod.visualization.helpers import get_transformed_3d_label_corners_cartesian


def locs_to_distance(locations: np.ndarray) -> np.ndarray:
    """
    Return the distance to the origin (0, 0, 0) for a given location array of shape (-1, 3)
    """
    
    return np.apply_along_axis(lambda row: np.linalg.norm(row), 1, locations)

def azimuth_angle_from_location(locations: np.ndarray) -> np.ndarray:
    """
    Return the azimuth angle of a location to the origin (0, 0) for a given locations array of shape (-1, 2)
    """
    
    # TODO test this function
    # TODO be sure we have the correct angle, i.e. counterclockwise/clockwise
    return np.apply_along_axis(lambda row: np.arctan2(row[0], row[1]), 1, locations)
   
def points_in_bbox(radar_points: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """
    Returns the radar points inside the given bounding box.
    Requires that radar points and bounding boxes are in the same coordinate system.

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

        # the correct bounding box shape can be seen in transformed_3d_labels!
        # first index see order of corners above        
        # second index is x, y, z of the corner
        if x >= bbox[2, 0] and x <= bbox[1, 0] and y >= bbox[1, 1] and y <= bbox[0, 1] and z >= bbox[0, 2] and z <= bbox[4, 2]:
            inside_points.append(radar_points[i])
            
    if not inside_points:
        return np.empty(0)
            
    return np.vstack(inside_points)
    
    
def dopplers_for_objects_in_frame(loader: FrameDataLoader, transforms: FrameTransformMatrix) -> np.ndarray:
    """
    For each object in the frame calculate its doppler value (if recognized).

    :param loader: the loader of the current frame
    :param transforms: the transformation matrix of the current frame

    Returns: a list of doppler values
    """
    
    # TODO we read the files this way twice, which is a bit suboptimal :O
    labels = FrameLabels(loader.get_labels())
    
    # Step 1: Obtain corners of bounding boxes and radar data points
    # TODO: is the last argument correct?
    
    # convert both to lidar coordinate system
    # we do not really care what coordinate system we use and this seems easier
    corners3d = get_transformed_3d_label_corners_cartesian(labels, transforms.t_camera_lidar, transforms.t_camera_lidar)
    
    # radar_points shape: [x, y, z, RCS, v_r, v_r_compensated, time] (-1, 7)
    radar_data = loader.radar_data
    radar_points = homogenous_transformation_cartesian_coordinates(radar_data[:, :3], transform=transforms.t_radar_lidar)
    radar_data_transformed = np.hstack((radar_points, loader.radar_data[:, 3:]))
    
    
    # Step 3: For each bounding box get a list of radar points which are inside of it
    dopplers = []
    for label in corners3d:
        bbox = label['corners_3d_transformed']
        radar_data_inside_bb = points_in_bbox(radar_points=radar_data_transformed, bbox=bbox)
        
        clazz = label['label_class']
        print(f'Class: {clazz}, Matches: {radar_data_inside_bb.shape[0]}')
        if radar_data_inside_bb.size != 0:
            # Step 4: Get the avg doppler value of the object and collect it
            doppler_mean = np.mean(radar_data_inside_bb[:, 4])
            dopplers.append(doppler_mean)
    
        
    if not dopplers:
        return np.empty(0)
    
    return np.vstack(dopplers)