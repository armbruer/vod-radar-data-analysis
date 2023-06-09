from typing import Dict, List
from vod.frame import FrameDataLoader
from vod.frame import FrameTransformMatrix
from vod.frame import FrameLabels
from vod.frame import homogenous_transformation_cartesian_coordinates
from vod.evaluation import evaluation_common as kitti
from vod.configuration.file_locations import KittiLocations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from vod.visualization.helpers import get_transformed_3d_label_corners_cartesian
    

def locs_to_distance(locations: List[np.ndarray]) -> List[np.ndarray]:
    """
    Return the distance to the origin (0, 0, 0) for a given list of locations of shape (-1, 3)
    """
    
    # every input np.ndarray is of shape (-1, 3)
    # x is a location vector of shape (3, )
    return map(lambda loc: np.apply_along_axis(lambda x: np.linalg.norm(x), 1, loc), locations)
   

# TODO maybe write test code for this?
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
    
    for i in radar_points.shape[0]:
        radar_point = radar_points[i, :3]
        x, y, z = radar_point
        
        # first index see order of corners above
        # second index is x, y, z of the corner
        if x >= bbox[2, 0] and x <= bbox[1, 0] and y >= bbox[1, 1] and y <= bbox[0, 1] and z >= bbox[0, 2] and z <= bbox[4, 2]:
            inside_points.append(radar_points[i])
            
    if not inside_points:
        return np.empty(0)
            
    return np.vstack(inside_points)
    
    
def dopplers_for_objects_in_frame(loader: FrameDataLoader, transforms: FrameTransformMatrix) -> List[np.ndarray]:
    # TODO we read the files this way twice, which is a bit suboptimal :O
    labels = FrameLabels(loader.get_labels)
    
    # Step 1: Obtain corners of bounding boxes and radar data points
    # TODO: is the last argument correct?
    corners3d = get_transformed_3d_label_corners_cartesian(labels, transforms.t_camera_radar, transforms.t_camera_lidar)
    
    # radar_points shape: [x, y, z, RCS, v_r, v_r_compensated, time] (-1, 7)
    radar_points = loader.radar_data() # in radar coordinates
    
    # Step 3: For each bounding box get a list of radar points which are inside of it
    dopplers = []
    for label in corners3d:
        bbox = label['corners_3d_transformed']
        radar_points_inside = points_in_bbox(radar_points=radar_points, bbox=bbox)
        
        if radar_points_inside.size != 0:
            # Step 4: Get the avg doppler value of the object and collect it
            doppler_mean = np.mean(radar_points_inside[:, 4])
            dopplers.append(doppler_mean)
    
    return np.vstack(dopplers)
    

def RAD_from_data(annotations: List[Dict], kitti_locations: KittiLocations) -> List[np.ndarray]:
    locations, azimuths, dopplers: List[np.ndarray] = [], [], []

    for anno in annotations:
        # operating here on a per frame basis!#
        frame_number = anno['frame_number']
        location_values: np.ndarray = anno['location'] # (-1, 3)
        azimuth_values: np.ndarray = anno['alpha']
        
        loader = FrameDataLoader(kitti_locations=kitti_locations, frame_number=frame_number)
        transforms = FrameTransformMatrix(frame_data_loader_object=loader)
        
        # Transform locations to cartesian coordinates
        locations_transformed = homogenous_transformation_cartesian_coordinates(location_values, transforms.t_radar_camera)
        # Calculate distance from location
        locations.append(locations_transformed)

        azimuths.append(azimuth_values)
        
        doppler_values: np.ndarray = dopplers_for_objects_in_frame(loader=loader, transforms=transforms) 
        dopplers.append(doppler_values)
    
    
    locations, azimuths,dopplers = map(np.vstack, [locations, azimuths, dopplers])
    azimuths = np.rad2deg(azimuths)
    ranges = locs_to_distance(locations)
    
    return [ranges, azimuths, dopplers]
        

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