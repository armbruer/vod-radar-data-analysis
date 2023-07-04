
from typing import List, Optional
from matplotlib.image import imsave

import numpy as np
from extraction.helpers import DataVariant, get_class_names, points_in_bbox
from vod.configuration.file_locations import KittiLocations
from vod.frame.data_loader import FrameDataLoader
from vod.frame.labels import FrameLabels
from vod.frame.transformations import FrameTransformMatrix, homogenous_transformation_cartesian_coordinates
from vod.visualization.helpers import get_placed_3d_label_corners
from vod.visualization.vis_2d import Visualization2D
from vod.visualization.vis_3d import Visualization3D


def visualize_frames(data_variant: DataVariant, 
                     kitti_locations: KittiLocations, 
                     frame_numbers: List[str], 
                     locs: List[np.ndarray]):
        """
        Visualizes radar points and corresponding annotations in 2D and 3D for the given frames (use semantic data_variants!).
        Useful for debugging.
        
        :param data_variant: the data variant for which to visualize the frame, 
                             depending on this different outputs will be generated,
                             use DataVariant.SEMANTIC_DATA if you want all outputs
        :param kitti_locations: the locations dirs
        :frame_numbers: the frame numbers to visualize
        :locs: the locations in the given frame_numbers to visualize
        
        """
        
        for frame_number, loc in zip(frame_numbers, locs):
            loader = FrameDataLoader(kitti_locations=kitti_locations, frame_number=frame_number)
            transforms = FrameTransformMatrix(frame_data_loader_object=loader)
            dv_str = data_variant.shortname()
            
            if data_variant in DataVariant.semantic_variants():
                vis2d = Visualization2D(frame_data_loader=loader, classes_visualized=get_class_names(summarized=False))
                
                labels = _find_labels_for_locs(loader, transforms, loc)
                
                _draw_helper2D(vis2d=vis2d, data_variant=data_variant, filename='radar')
                _draw_helper2D(vis2d=vis2d, data_variant=data_variant, filename='extremum-highlighted', selected_points=loc, selected_labels=labels)
                
                vis3d = Visualization3D(loader, origin='camera')
                _draw_helper3D(vis3d=vis3d, data_variant=data_variant, filename='extremum-highlighted', selected_points=loc, selected_labels=labels)

            imsave(f'{kitti_locations.analysis_dir}/{dv_str}/{frame_number}.png', loader.image)


def _find_labels_for_locs(loader: FrameDataLoader, transforms: FrameTransformMatrix, locs_radar: np.ndarray) -> Optional[FrameLabels]:
        # there is probably a more efficient way to do this whole method, but time
        labels = loader.get_labels()
        if labels is None:
            return None
        
        labels = get_placed_3d_label_corners(FrameLabels(labels))
        radar_data = loader.radar_data
        if radar_data is None:
            return None
        
        locs_camera = homogenous_transformation_cartesian_coordinates(locs_radar, transform=transforms.t_camera_radar)
        
        matching_labels = [label for label in labels 
                           if points_in_bbox(radar_points_radar=locs_radar, radar_points_camera=locs_camera, bbox=label['corners_3d_placed']) is not None]
        res = FrameLabels([]) # a bit hacky, but do not set the raw labels
        res._labels_dict = matching_labels
        return res

     
def _draw_helper2D(vis2d: Visualization2D,
                  data_variant: DataVariant, 
                  filename: str, 
                  lidar=False,
                  selected_points: Optional[np.ndarray]=None,
                  selected_labels: Optional[FrameLabels]=None):
        dv_str = data_variant.shortname()
        
        vis2d.draw_plot(plot_figure=False, 
                        save_figure=True, 
                        show_gt=True,
                        show_lidar=lidar, 
                        show_radar=True, 
                        subdir=f'analysis/{dv_str}', 
                        filename=f'{dv_str}-{filename}',
                        selected_points=selected_points,
                        selected_labels=selected_labels,
                        max_distance_threshold=105,
                        min_distance_threshold=-10)
        
def _draw_helper3D(
                vis3d: Visualization3D, 
                data_variant: DataVariant, 
                filename: str, 
                selected_points: Optional[np.ndarray]=None,
                selected_labels: Optional[FrameLabels]=None):
    dv_str = data_variant.shortname()
    
    vis3d.draw_plot(radar_origin_plot=True,
                    camera_origin_plot=True,
                    radar_points_plot=True,
                    annotations_plot=True,
                    write_to_html=True,
                    html_name=f'{dv_str}-{filename}',
                    subdir=f'analysis/{dv_str}',
                    selected_points=selected_points,
                    selected_labels=selected_labels,
                    auto_frame=True,
                    grid_visible=True)