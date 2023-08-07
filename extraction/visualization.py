import logging
import pathlib
import numpy as np

from typing import List, Optional
from matplotlib.image import imsave

from extraction.helpers import DataVariant, find_matching_points_for_bboxes, get_class_names, prepare_radar_data
from vod.configuration.file_locations import KittiLocations
from vod.frame.data_loader import FrameDataLoader
from vod.frame.labels import FrameLabels
from vod.frame.transformations import FrameTransformMatrix, homogenous_transformation_cart
from vod.visualization.vis_2d import Visualization2D
from vod.visualization.vis_3d import Visualization3D



def visualize_frame_sequence(
                    data_variant: DataVariant,
                    kitti_locations: KittiLocations,
                    min_frame_number: int,
                    max_frame_number: int, 
                    frame_labels: Optional[List[FrameLabels]]=None,
                    tracking_id: Optional[int]=None):
    assert max_frame_number > min_frame_number
    
    frame_numbers = list(range(min_frame_number, max_frame_number+1))
    frame_numbers = map(lambda fn: str(fn).zfill(5), frame_numbers)
    
    visualize_frames(data_variant=data_variant, 
                     kitti_locations=kitti_locations, 
                     frame_numbers=frame_numbers, 
                     frame_labels=frame_labels,
                     tracking_id=tracking_id)


def visualize_frames(data_variant: DataVariant, 
                     kitti_locations: KittiLocations,
                     frame_numbers: List[str],
                     tracking_id: Optional[int]=None,
                     frame_labels: Optional[List[FrameLabels]]=None):
    
    if frame_labels is None:
        # if we don't already get labels, lets just load them from file
        frame_labels: List[FrameLabels] = []
        for frame_number in frame_numbers:
            loader = FrameDataLoader(kitti_locations=kitti_locations, frame_number=frame_number)
            labels = loader.get_labels()
            if labels is None:
                logging.error("One of the frame numbers has no labels")
                return
            
            frame_labels.append(labels)

    if tracking_id is not None:
        # keep only those labels and frame_numbers that match the tracking id given
        only_tracking = lambda labels: filter(lambda l: l['tracking_id'] == tracking_id, labels)
        
        frame_labels = [only_tracking(labels) for labels in frame_labels]
        
        to_be_removed = []
        for i, (fn, fl) in enumerate(zip(frame_numbers, frame_labels)):
            if not fl:
                logging.info(f"Frame number {fn} has no object with tracking id {tracking_id}. \
                             No output will be created for this frame_number!")
            
            to_be_removed.append(i)
        
        del frame_labels[to_be_removed]
        del frame_numbers[to_be_removed]
        
    for fn, fl in zip(frame_numbers, frame_labels):
        visualize_frame(data_variant=data_variant, kitti_locations=kitti_locations, frame_number=fn, frame_labels=fl)
        

def visualize_frame(data_variant: DataVariant, 
                     kitti_locations: KittiLocations, 
                     frame_number: str,
                     center_radar: np.ndarray,
                     detections: int = -1, 
                     i: int = 0,
                     runs_counter: int = 0,
                     frame_labels: Optional[FrameLabels]=None):
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
    center_radar = np.atleast_2d(center_radar).astype(np.float32)
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
                        matching_points=center_radar, show_gt=False)
        
        _visualize3D(vis3d=vis3d, subdir=subdir, filename=filename_extremum, 
                        matching_points=center_radar, annotations=False)
        return
    
    transforms = FrameTransformMatrix(loader)
    if frame_labels is None:
        labels: Optional[FrameLabels] = _find_matching_labels(loader=loader, transforms=transforms, center_radar=center_radar)
        if labels is None:
            return
    else:
        labels = frame_labels
    
    matching_points: Optional[np.ndarray] = _find_matching_points(labels=labels, loader=loader, transforms=transforms, detections=detections)
    if matching_points is None:
        return
    
    # there is a 2D bounding box visualization bug that is part of the original VoD code
    # I double checked whether the bbox in this frame is visualized also wrongly in the original code
    # An example of this bug can be seen in frame 01839
    
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
                        detections: int) -> Optional[List[np.ndarray]]:
    radar_data_r = prepare_radar_data(loader)
    if radar_data_r is None:
        return None
    
    matching_points = find_matching_points_for_bboxes(radar_points=radar_data_r, labels=labels, transforms=transforms)
    
    # debugging
    for points in matching_points:
        if points is None:
            # this indicates an error, but is not handled as such, as I want to continue execution
            logging.error("No matching points :(")

        if detections != -1 and len(points) != detections:
            logging.error("Detections did not match!")
    
    matching_points = map(lambda p: p[1], matching_points)
    matching_points = filter(lambda p: p is not None, matching_points)
    if not matching_points:
        return None
    
    return np.vstack(matching_points)

     
def _visualize2D(vis2d: Visualization2D,
                  subdir: str,
                  filename: str, 
                  lidar=False,
                  matching_points: Optional[np.ndarray]=None,
                  selected_labels: Optional[FrameLabels]=None, 
                  show_gt=True):
        
        vis2d.draw_plot(plot_figure=False, 
                        save_figure=True, 
                        show_gt=show_gt,
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
                selected_labels: Optional[FrameLabels]=None, 
                annotations: bool=True):

    
    vis3d.draw_plot(radar_origin_plot=True,
                    camera_origin_plot=True,
                    radar_points_plot=True,
                    annotations_plot=annotations,
                    write_to_html=True,
                    html_name=filename,
                    subdir=subdir,
                    selected_points=matching_points,
                    selected_labels=selected_labels,
                    auto_frame=True,
                    grid_visible=True,
                    radar_velocity_plot=annotations)
    