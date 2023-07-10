
from typing import List, Optional
from matplotlib.image import imsave

import numpy as np
from extraction.helpers import DataVariant, get_class_names, points_in_bbox
from vod.configuration.file_locations import KittiLocations
from vod.frame.data_loader import FrameDataLoader
from vod.frame.labels import FrameLabels
from vod.frame.transformations import FrameTransformMatrix, homogenous_transformation_cart
from vod.visualization.helpers import get_placed_3d_label_corners
from vod.visualization.vis_2d import Visualization2D
from vod.visualization.vis_3d import Visualization3D
import logging


def visualize_frames(data_variant: DataVariant, 
                     kitti_locations: KittiLocations, 
                     frame_numbers: List[str], 
                     locs: List[np.ndarray],
                     detections: List[int]):
        """
        Visualizes radar points and corresponding annotations in 2D and 3D for the given frames (use semantic data_variants!).
        Useful for debugging.
        
        :param data_variant: the data variant for which to visualize the frame, 
                             depending on this different outputs will be generated,
                             use DataVariant.SEMANTIC_DATA if you want all outputs
        :param kitti_locations: the locations dirs
        :frame_numbers: the frame numbers to visualize
        :locs: the locations in the given frame_numbers to visualize (in radar coordinates)
        
        """
        
        for frame_number, loc_radar, dets in zip(frame_numbers, locs, detections):
            loader = FrameDataLoader(kitti_locations=kitti_locations, frame_number=frame_number)
            dv_str = data_variant.shortname()
            
            if data_variant in DataVariant.semantic_variants():
                vis2d = Visualization2D(frame_data_loader=loader, classes_visualized=get_class_names(summarized=False))
                
                labels, matching_points = _find_labels_and_points_for_center(loader, loc_radar, dets)
                
                # there is a 2D bounding box visualization bug that is part of the original VoD code
                # I double checked whether the bbox in this frame is visualized also wrongly in the original code
                # An example of this bug can be seen in frame 01839
                # the bug is not relevant for us, we can use different samples for visualization in the thesis
                _draw_helper2D(vis2d=vis2d, data_variant=data_variant, filename='radar')
                _draw_helper2D(vis2d=vis2d, data_variant=data_variant, filename='extremum-highlighted', matching_points=matching_points, selected_labels=labels)
                
                vis3d = Visualization3D(loader, origin='camera')
                _draw_helper3D(vis3d=vis3d, data_variant=data_variant, filename='radar')
                _draw_helper3D(vis3d=vis3d, data_variant=data_variant, filename='extremum-highlighted', matching_points=matching_points, selected_labels=labels)
            else:
                imsave(f'{kitti_locations.analysis_dir}/{dv_str}/{frame_number}.png', loader.image)


def _find_labels_and_points_for_center(loader: FrameDataLoader, 
                                       center_radar: np.ndarray, 
                                       detections: int) -> Optional[FrameLabels]:
        # 1. Find the labels_dict matching the center point (that means the object annotation corresponding to the center)
        labels = loader.get_labels()
        if labels is None:
            return None
        
        labels = FrameLabels(labels)
        
        transforms = FrameTransformMatrix(loader)
        center_camera = homogenous_transformation_cart(center_radar, transforms.t_camera_radar)
        
        x, y, z = center_camera[0, :3]
        labels_matching_objects = []
        for label in labels.labels_dict:
            # we have lost somewhere a bit of precision...probs while storing stuff
            if np.isclose(label['x'], x) and np.isclose(label['y'], y) and np.isclose(label['z'],z):
                labels_matching_objects.append(label)
        if len(labels_matching_objects) == 0:
            logging.error("No object matching center point found")
        if len(labels_matching_objects) > 1:
            # not sure if this case exists in the data (e.g. for rider class, but unlikely)
            logging.warning("More than one object matching the center found")
        
        resLabels = FrameLabels([]) # a bit hacky, but do not set the raw labels
        resLabels._labels_dict = labels_matching_objects
        
        # 2. Find the radar points we originally said are inside of this bbox
        # We need to draw them again to visually verify the correctness of our methods
        
        # we apply the same matching algorithm for points as in extract.py
        radar_data_r = loader.radar_data
        if radar_data_r is None:
            return None
        
        radar_data_r = radar_data_r[np.where(radar_data_r[:, 6] == 0)]
        
        transforms = FrameTransformMatrix(loader)
        labels_3d_corners = get_placed_3d_label_corners(resLabels, transforms)
        assert len(labels_3d_corners) == 1
        
        bbox = labels_3d_corners[0]['corners_3d_placed']
        matching_points_r = points_in_bbox(radar_points=radar_data_r, bbox=bbox)
        if matching_points_r is None:
            # this indicates an error, but is not handled as such, as I want to continue execution
            logging.error("No matching points :(")
            return resLabels, None
        
        if detections != len(matching_points_r):
            logging.error("Detections did not match!")
            
        
        return resLabels, np.vstack(matching_points_r)

     
def _draw_helper2D(vis2d: Visualization2D,
                  data_variant: DataVariant, 
                  filename: str, 
                  lidar=False,
                  matching_points: Optional[np.ndarray]=None,
                  selected_labels: Optional[FrameLabels]=None):
        dv_str = data_variant.shortname()
        
        vis2d.draw_plot(plot_figure=False, 
                        save_figure=True, 
                        show_gt=True,
                        show_lidar=lidar, 
                        show_radar=True, 
                        subdir=f'analysis/{dv_str}', 
                        filename=f'{dv_str}-{filename}',
                        selected_points=matching_points,
                        selected_labels=selected_labels,
                        max_distance_threshold=105,
                        min_distance_threshold=-10)
        
def _draw_helper3D(
                vis3d: Visualization3D, 
                data_variant: DataVariant, 
                filename: str, 
                matching_points: Optional[np.ndarray]=None,
                selected_labels: Optional[FrameLabels]=None):
    dv_str = data_variant.shortname()
    
    vis3d.draw_plot(radar_origin_plot=True,
                    camera_origin_plot=True,
                    radar_points_plot=True,
                    annotations_plot=True,
                    write_to_html=True,
                    html_name=f'{dv_str}-{filename}',
                    subdir=f'analysis/{dv_str}',
                    selected_points=matching_points,
                    selected_labels=selected_labels,
                    auto_frame=True,
                    grid_visible=True,
                    radar_velocity_plot=True)