from typing import Optional
import numpy as np

from vod.frame.data_loader import FrameDataLoader
from vod.frame.labels import FrameLabels
from vod.frame.transformations import FrameTransformMatrix, homogenous_transformation_cartesian_coordinates
from vod.visualization.helpers import get_placed_3d_label_corners, get_transformed_3d_label_corners_cartesian


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
   
def points_in_bbox(radar_points: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
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
        return None
            
    return np.vstack(inside_points)

def get_class_list():
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

def name_from_class_id(clazz: int) -> str:
    
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
    
    
def class_id_from_name(name: str) -> int:
    
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
    
    
def get_data_for_objects_in_frame(loader: FrameDataLoader, transforms: FrameTransformMatrix) -> Optional[np.ndarray]:
    """
    For each object in the frame retrieve the following data: object tracking id, object class, absolute velocity, 
    number of detections, bounding box volume, ranges, azimuths, relative velocity (doppler).

    :param loader: the loader of the current frame
    :param transforms: the transformation matrix of the current frame

    Returns: a numpy array with the following columns: object tracking id, object class, absolute velocity, 
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
    
    radar_points = homogenous_transformation_cartesian_coordinates(radar_data[:, :3], transform=transforms.t_camera_radar)
    radar_data_transformed = np.hstack((radar_points, loader.radar_data[:, 3:]))
    
    
    # Step 3: For each bounding box get a list of radar points which are inside of it
    object_ids = [] # TODO
    object_clazz = []
    velocity_abs = [] # one avg absolute velocity per bounding box
    dopplers = [] # one avg doppler value per bounding box
    detections = [] # number of radar_points inside a bounding box
    bbox_vols = [] # bounding box volume
    ranges = [] # range in m
    azimuths = [] # azimuth in degree
    
    
    for label in labels_with_corners:
        bbox = label['corners_3d_placed']
        points_matching = points_in_bbox(radar_points=radar_data_transformed, bbox=bbox)
        
        if points_matching is not None:
            class_name = label['label_class']
            # Step 4: Get the avg doppler value of the object and collect it
            
            object_clazz.append(class_id_from_name(class_name))
            velocity_abs.append(np.mean(points_matching[:, 5]))
            detections.append(points_matching.shape[0])
            bbox_vols.append(label['l'] * label['h'] * label['w'])            
            
            loc = np.array([[label['x'], label['y'], label['z']]])
            loc_transformed = homogenous_transformation_cartesian_coordinates(loc, transforms.t_radar_camera)
            range_from_loc = locs_to_distance(loc_transformed)
            ranges.append(range_from_loc)
            
            azimuths.append(np.rad2deg(azimuth_angle_from_location(np.array([[label['x'], label['y']]]))))
            dopplers.append(np.mean(points_matching[:, 4]))
    
    if not object_clazz:
        return None
    
    columns = [object_clazz, velocity_abs, detections, bbox_vols, ranges, azimuths, dopplers]
    # create one array of shape (-1, 7)
    return np.vstack(list(map(np.hstack, columns))).T

def vis_to_debug(frame_data: FrameDataLoader):
    from vod.visualization import Visualization3D
    vis3d = Visualization3D(frame_data)

    vis3d.draw_plot(radar_origin_plot = True,
                  lidar_origin_plot = True,
                  camera_origin_plot = True,
                  lidar_points_plot = True,
                  radar_points_plot = True,
                  annotations_plot = True)
    
    from vod.visualization import Visualization2D
    vis2d = Visualization2D(frame_data)
    
    vis2d.draw_plot(show_lidar=True,
                             show_radar=True,
                             show_gt=True,
                             min_distance_threshold=5,
                             max_distance_threshold=40,
                             save_figure=True)