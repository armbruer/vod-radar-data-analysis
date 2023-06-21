
import logging
import os
from typing import List
from matplotlib.image import imsave
import numpy as np
import pandas as pd

from extraction.file_manager import DataManager
from extraction.helpers import DataVariant, get_class_list
from vod.configuration.file_locations import KittiLocations

from vod.frame.data_loader import FrameDataLoader
from vod.visualization.vis_2d import Visualization2D

class DataAnalysisHelper:
    
    def __init__(self, data_manager: DataManager) -> None:
        self.data_manager = data_manager
        self.kitti_locations = data_manager.kitti_locations

    def prepare_data_analysis(self, data_variant: DataVariant):
        df = self.data_manager.get_df(data_variant=data_variant)
        
        def framenums_from_index(indexes: np.ndarray, data: np.ndarray):
            return data[indexes, 0]
        
        data = df.to_numpy()
        
        mins = np.round(np.min(data[:, 1:], axis=0).astype(np.float64), decimals=2)
        min_fns = framenums_from_index(np.argmin(data[:, 1:], axis=0), data)
        
        maxs = np.round(np.max(data[:, 1:], axis=0).astype(np.float64), decimals=2)
        max_fns = framenums_from_index(np.argmax(data[:, 1:], axis=0), data)
        
        dv_str = data_variant.name.lower()
        dir = f'{self.kitti_locations.analysis_dir}/{dv_str}'
        os.makedirs(dir, exist_ok=True)
        
        for min_fn, max_fn in zip(min_fns, max_fns):
            self._visualize_frames(data_variant=data_variant, kitti_locations=self.kitti_locations, frame_numbers=[min_fn, max_fn])
            
        stats = np.vstack((mins, min_fns, maxs, max_fns))
        columns = list(map(lambda c: c.capitalize(), list(df.columns)[1:]))

        df = pd.DataFrame(stats, columns=columns)
        df.insert(0, "Name", pd.Series(["Min", "Min Frame Number", "Max", "Max Frame Number"]))

        filename = f'{dir}/{dv_str}.csv'
        
        df.to_csv(filename, index=False)

        logging.info(f'Analysis data written to file:///{filename}.csv')
        
    
    def _visualize_frames(self, data_variant: DataVariant, kitti_locations: KittiLocations, frame_numbers: List[str]):
        for frame_number in frame_numbers:
            loader = FrameDataLoader(kitti_locations=kitti_locations, frame_number=frame_number)
            dv_str = data_variant.name.lower()
            
            if data_variant not in [DataVariant.SYNTACTIC_RAD, DataVariant.STATIC_DYNAMIC_RAD]:
                vis2d = Visualization2D(frame_data_loader=loader, classes_visualized=get_class_list())
                vis2d.draw_plot(plot_figure=False, save_figure=True, show_gt=True,
                        show_lidar=True, show_radar=True, outdir=f'analysis/{dv_str}/annotated-') # TODO
                
                vis2d.draw_plot(plot_figure=False, save_figure=True, show_gt=True,
                        show_lidar=False, show_radar=True, outdir=f'analysis/{dv_str}/radar-annotated-') # TODO

            imsave(f'{kitti_locations.analysis_dir}/{dv_str}/{frame_number}.png', loader.image)


def prepare_data_analysis(data_manager: DataManager):
    dvs = [DataVariant.SYNTACTIC_RAD, DataVariant.STATIC_DYNAMIC_RAD,
           DataVariant.SEMANTIC_OBJECT_DATA, DataVariant.SEMANTIC_OBJECT_DATA_BY_CLASS]

    analysis = DataAnalysisHelper(data_manager)
    
    
    for dv in dvs:
        analysis.prepare_data_analysis(dv)