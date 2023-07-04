import logging
import os
import numpy as np
import pandas as pd

from typing import List, Optional
from matplotlib.image import imsave
from extraction.file_manager import DataManager, DataView
from extraction.helpers import DataVariant, DataViewType, get_class_names, points_in_bbox

from vod.configuration.file_locations import KittiLocations
from vod.frame.data_loader import FrameDataLoader
from vod.frame.labels import FrameLabels
from vod.frame.transformations import FrameTransformMatrix, homogenous_transformation_cartesian_coordinates
from vod.visualization.helpers import get_placed_3d_label_corners
from vod.visualization.vis_2d import Visualization2D
from vod.visualization.vis_3d import Visualization3D

class DataAnalysisHelper:
    
    def __init__(self, data_manager: DataManager) -> None:
        self.data_manager = data_manager
        self.kitti_locations = data_manager.kitti_locations
        
    def prepare_data_analysis(self, data_variant: DataVariant, data_view: DataViewType):
        data_view: DataView = self.data_manager.get_view(data_variant=data_variant, data_view_type=data_view)
        df = data_view.df
        if isinstance(df, list):
            for d, v in zip(df, data_variant.subvariant_names()):
                self._prepare_data_analysis( d, data_variant, v)
            return
        
        self._prepare_data_analysis(df, data_variant)
        
    def _framenums_from_index(self, indexes: np.ndarray, data: np.ndarray) -> List[int]:
        return list(data[indexes, 0]) # 0 is framenumber
    
    def _loc_from_index(self, indexes: np.ndarray, data: np.ndarray) -> List[np.ndarray]:
        locs: List[np.ndarray] = list(data[indexes, -3:]) # the last three are x, y, z
        
        locs = map(lambda loc: loc[np.newaxis, ...] if loc.ndim == 1 else loc, locs)
        locs = [loc.astype(np.float64) for loc in locs]
        
        return locs

    def _prepare_data_analysis(self, df: pd.DataFrame, data_variant: DataVariant, subvariant: str = ''):
        data = df.to_numpy()
        rdata = data[:, 1:-3] # exclude the last three, they are x y z and the first one, its the frame number
        
        mins = np.round(np.min(rdata, axis=0).astype(np.float64), decimals=2)
        min_indexes = np.argmin(rdata, axis=0)
        min_fns = self._framenums_from_index(min_indexes, data)
        min_locs = self._loc_from_index(min_indexes, data)
        
        maxs = np.round(np.max(rdata, axis=0).astype(np.float64), decimals=2)
        max_indexes = np.argmax(rdata, axis=0)
        max_fns = self._framenums_from_index(max_indexes, data)
        max_locs = self._loc_from_index(max_indexes, data)
        
        dv_str = data_variant.shortname()
        dir = f'{self.kitti_locations.analysis_dir}/{dv_str}'
        dir = dir if not subvariant else f'{dir}/{subvariant}'
        os.makedirs(dir, exist_ok=True)
        
        iter = zip(min_fns, max_fns, min_locs, max_locs)
        for min_fn, max_fn, min_loc, max_loc in iter:
            self._visualize_frames(data_variant=data_variant, kitti_locations=self.kitti_locations, frame_numbers=[min_fn, max_fn], locs=[min_loc, max_loc])
            
        stats = np.vstack((mins, min_fns, maxs, max_fns))
        columns = list(map(lambda c: c.capitalize(), list(df.columns)[1:-3]))

        df = pd.DataFrame(stats, columns=columns)
        df.insert(0, "Name", pd.Series(["Min", "Min Frame Number", "Max", "Max Frame Number"]))
        
        filename = f'{dir}/{dv_str}.csv'
        
        df.to_csv(filename, index=False)

        logging.info(f'Analysis data written to file:///{filename}')
        
    def _find_labels_for_locs(self, loader: FrameDataLoader, transforms: FrameTransformMatrix, locs_radar: np.ndarray) -> Optional[FrameLabels]:
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
    
    
    def _draw_helper2D(self,
                     vis2d: Visualization2D, 
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
        
    def _draw_helper3D(self,
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
                        selected_labels=selected_labels)
        
                
    
    def _visualize_frames(self, 
                          data_variant: DataVariant, 
                          kitti_locations: KittiLocations, 
                          frame_numbers: List[str], 
                          locs: List[np.ndarray]):
        
        for frame_number, loc in zip(frame_numbers, locs):
            loader = FrameDataLoader(kitti_locations=kitti_locations, frame_number=frame_number)
            transforms = FrameTransformMatrix(frame_data_loader_object=loader)
            dv_str = data_variant.shortname()
            
            if data_variant in DataVariant.semantic_variants():
                vis2d = Visualization2D(frame_data_loader=loader, classes_visualized=get_class_names(summarized=False))
                
                labels = self._find_labels_for_locs(loader, transforms, loc)
                
                self._draw_helper2D(vis2d=vis2d, data_variant=data_variant, filename='radar')
                self._draw_helper2D(vis2d=vis2d, data_variant=data_variant, filename='extremum-highlighted', selected_points=loc, selected_labels=labels)
                
                vis3d = Visualization3D(loader, origin='camera') # TODO camera?
                self._draw_helper3D(vis3d=vis3d, data_variant=data_variant, filename='extremum-highlighted', selected_points=loc, selected_labels=labels)

            imsave(f'{kitti_locations.analysis_dir}/{dv_str}/{frame_number}.png', loader.image)


def prepare_data_analysis(data_manager: DataManager):
    analysis = DataAnalysisHelper(data_manager)
    
    for dv in DataVariant.all_variants():
        analysis.prepare_data_analysis(dv, DataViewType.BASIC_ANALYSIS)
        #analysis.prepare_data_analysis(dv, DataViewType.ANALYSIS)