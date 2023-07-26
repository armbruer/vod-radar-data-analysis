import logging
import pathlib
import numpy as np

from typing import Optional
from matplotlib.image import imsave
from extraction.helpers import DataVariant, find_matching_points_for_bboxes, get_class_names, prepare_radar_data
from vod.configuration.file_locations import KittiLocations
from vod.frame.data_loader import FrameDataLoader
from vod.frame.labels import FrameLabels
from vod.frame.transformations import FrameTransformMatrix, homogenous_transformation_cart
from vod.visualization.vis_2d import Visualization2D
from vod.visualization.vis_3d import Visualization3D


def visualize_frame(data_variant: DataVariant, 
                     kitti_locations: KittiLocations, 
                     frame_number: str,
                     center_radar: np.ndarray,
                     detections: int, 
                     i=0,
                     runs_counter=0):
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
    loader = FrameDataLoader(kitti_locations=kitti_locations, frame_number=frame_number)
    dv_str = data_variant.shortname()
    filename_extremum = f'{dv_str}-extremum-highlighted-{i}-{runs_counter}'
    dir = f'{kitti_locations.analysis_dir}'
    subdir: str = f'{pathlib.Path(dir).name}/{dv_str}'
    
    vis2d = Visualization2D(frame_data_loader=loader, 
                            classes_visualized=get_class_names(summarized=False))
    
    vis3d = Visualization3D(loader, origin='camera')
    
    if data_variant not in DataVariant.semantic_variants():
        imsave(f'{dir}/{dv_str}/{frame_number}.png', loader.image)
        
        # center radar is here simply the extremum
        _visualize2D(vis2d=vis2d, subdir=subdir, filename=filename_extremum, 
                        matching_points=center_radar)
        
        _visualize3D(vis3d=vis3d, subdir=subdir, filename=filename_extremum, 
                        matching_points=center_radar)
        return
    
    transforms = FrameTransformMatrix(loader)
    labels: Optional[FrameLabels] = _find_matching_labels(loader=loader, transforms=transforms, center_radar=center_radar)
    if labels is None:
        return
    
    matching_points: Optional[np.ndarray] = _find_matching_points(labels=labels, loader=loader, transforms=transforms, detections=detections)
    if matching_points is None:
        return
    
    # there is a 2D bounding box visualization bug that is part of the original VoD code
    # I double checked whether the bbox in this frame is visualized also wrongly in the original code
    # An example of this bug can be seen in frame 01839
    # the bug is not relevant for us, we can use different samples for visualization in the thesis
    
    filename_radar = f'{dv_str}-radar-{i}-{runs_counter}'
    vis2d = Visualization2D(frame_data_loader=loader, 
                            classes_visualized=get_class_names(summarized=False))
    _visualize2D(vis2d=vis2d, subdir=subdir, filename=filename_radar)
    _visualize2D(vis2d=vis2d, subdir=subdir, filename=filename_extremum, 
                    matching_points=matching_points, selected_labels=labels)
    
    _visualize3D(vis3d=vis3d, subdir=subdir, filename=filename_radar)
    _visualize3D(vis3d=vis3d, subdir=subdir, filename=filename_extremum, 
                    matching_points=matching_points, selected_labels=labels)


def _find_matching_labels(loader: FrameDataLoader, 
                          transforms: FrameTransformMatrix, 
                          center_radar: np.ndarray) -> Optional[FrameLabels]:
    # Find the labels_dict matching the center point 
    # (that means the object annotation corresponding to the center)
    labels = loader.get_labels()
    if labels is None:
        return None
    
    labels = FrameLabels(labels)
    
    center_radar = np.atleast_2d(center_radar)
    center_camera = homogenous_transformation_cart(points=center_radar, transform=transforms.t_camera_radar)
    x, y, z = center_camera[0]
    
    labels_matching_objects = []
    for label in labels.labels_dict:
        # we have lost somewhere a bit of precision...probs while storing stuff
        if np.isclose(label['x'], x) and np.isclose(label['y'], y) and np.isclose(label['z'],z):
            labels_matching_objects.append(label)
    
    if len(labels_matching_objects) == 0:
        logging.error("No object matching center point found")
        return None
    
    if len(labels_matching_objects) > 1:
        # not sure if this case exists in the data (e.g. for rider class, but unlikely)
        logging.warning("More than one object matching the center found")
    
    resLabels = FrameLabels([]) # a bit hacky, but do not set the raw labels
    resLabels._labels_dict = labels_matching_objects
    return resLabels

def _find_matching_points(labels: FrameLabels, 
                        loader: FrameDataLoader, 
                        transforms: FrameTransformMatrix,
                        detections: int) -> Optional[np.ndarray]:
    radar_data_r = prepare_radar_data(loader)
    if radar_data_r is None:
        return None
    
    assert len(labels._labels_dict) == 1
    
    _, matching_points = find_matching_points_for_bboxes(radar_points=radar_data_r, 
                                                            labels=labels, transforms=transforms)[0]
    if matching_points is None:
        # this indicates an error, but is not handled as such, as I want to continue execution
        logging.error("No matching points :(")
        return None

    if len(matching_points) != detections:
        logging.error("Detections did not match!")
    
    return matching_points

     
def _visualize2D(vis2d: Visualization2D,
                  subdir: str,
                  filename: str, 
                  lidar=False,
                  matching_points: Optional[np.ndarray]=None,
                  selected_labels: Optional[FrameLabels]=None):
        
        vis2d.draw_plot(plot_figure=False, 
                        save_figure=True, 
                        show_gt=True,
                        show_lidar=lidar, 
                        show_radar=True, 
                        subdir=subdir, 
                        filename=filename,
                        selected_points=matching_points,
                        selected_labels=selected_labels,
                        max_distance_threshold=105,
                        min_distance_threshold=-10)
        
def _visualize3D(
                vis3d: Visualization3D, 
                subdir: str,
                filename: str, 
                matching_points: Optional[np.ndarray]=None,
                selected_labels: Optional[FrameLabels]=None):

    
    vis3d.draw_plot(radar_origin_plot=True,
                    camera_origin_plot=True,
                    radar_points_plot=True,
                    annotations_plot=True,
                    write_to_html=True,
                    html_name=filename,
                    subdir=subdir,
                    selected_points=matching_points,
                    selected_labels=selected_labels,
                    auto_frame=True,
                    grid_visible=True,
                    radar_velocity_plot=True)
    