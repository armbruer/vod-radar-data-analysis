from typing import Dict, List

# TODO
import sys
sys.path.append("/home/eric/Documents/mt/radar_dataset/view-of-delft-dataset")

from vod.frame import FrameDataLoader
from vod.frame import FrameTransformMatrix
from vod.frame import FrameLabels
from vod.frame import homogenous_transformation_cartesian_coordinates
from vod.visualization.helpers import get_transformed_3d_label_corners_cartesian

from vod.evaluation import evaluation_common as kitti
from vod.configuration.file_locations import KittiLocations

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
    

def locs_to_distance(locations: List[np.ndarray]) -> List[np.ndarray]:
    """
    Return the distance to the origin (0, 0, 0) for a given list of locations of shape (-1, 3)
    """
    
    # every input np.ndarray is of shape (-1, 3)
    # x is a location vector of shape (3, )
    return map(lambda loc: np.apply_along_axis(lambda x: np.linalg.norm(x), 1, loc), locations)
   

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
    
    
def dopplers_for_objects_in_frame(loader: FrameDataLoader, transforms: FrameTransformMatrix) -> List[np.ndarray]:
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
    

def RAD_from_data(annotations: List[Dict], kitti_locations: KittiLocations) -> List[np.ndarray]:
    """Get the Range, Azimuth, Doppler values for each frame and object in this dataset.
    
    :param annotations: the list of label annotations in this dataset
    :param kitti_locations: the KittiLocation of this dataset
    
    Returns a list of range, azimuth, doppler arrays
    """
    
    locations: List[np.ndarray] = []
    azimuths: List[np.ndarray] = []
    dopplers: List[np.ndarray] = []

    for anno in annotations:
        # operating here on a per frame basis!#
        frame_number = anno['frame_number']
        location_values: np.ndarray = anno['location'] # (-1, 3)
        azimuth_values: np.ndarray = anno['alpha']
        
        loader = FrameDataLoader(kitti_locations=kitti_locations, frame_number=frame_number)
        transforms = FrameTransformMatrix(frame_data_loader_object=loader)
        
        # Transform locations to cartesian coordinates
        locations_transformed = homogenous_transformation_cartesian_coordinates(location_values, transforms.t_camera_radar)
        locations.append(locations_transformed)

        azimuths.append(azimuth_values)
        
        doppler_values: np.ndarray = dopplers_for_objects_in_frame(loader=loader, transforms=transforms) 
        dopplers.append(doppler_values)
    
    
    locations, azimuths,dopplers = map(np.vstack, [locations, azimuths, dopplers])
    azimuths = np.rad2deg(azimuths)
    ranges = locs_to_distance(locations)
    
    return [ranges, azimuths, dopplers]


def main():
    output_dir = "output"
    root_dir = "/home/eric/Documents/mt/radar_dataset/view_of_delft_PUBLIC/"
    kitti_locations = KittiLocations(root_dir=root_dir,
                                    output_dir="output",
                                    frame_set_path="",
                                    pred_dir="",
                                    )

    print(f"Lidar directory: {kitti_locations.lidar_dir}")
    print(f"Radar directory: {kitti_locations.radar_dir}")


    dt_annotations = kitti.get_label_annotations(kitti_locations.label_dir)
    rad = RAD_from_data(annotations=dt_annotations, kitti_locations=kitti_locations)


    columns = ["range (m)", "angle (degree)", "doppler (m/s)"]
    fig, axs = plt.subplots(1, 3)

    for i, column in enumerate(columns):
        sns.violinplot(x=rad[i], ax=axs[i])
        axs[i].set_title(column)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()