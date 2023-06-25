import logging
import os
import numpy as np
import pandas as pd

from typing import List
from matplotlib.image import imsave
from datetime import datetime
from extraction.file_manager import DataManager
from extraction.helpers import DataVariant, DataView, get_class_names

from vod.configuration.file_locations import KittiLocations
from vod.frame.data_loader import FrameDataLoader
from vod.visualization.vis_2d import Visualization2D

class DataAnalysisHelper:
    
    def __init__(self, data_manager: DataManager) -> None:
        self.data_manager = data_manager
        self.kitti_locations = data_manager.kitti_locations
        
    def prepare_data_analysis(self, data_variant: DataVariant, data_view: DataView):
        df = self.data_manager.get_df(data_variant=data_variant, data_view=data_view)
        if isinstance(df, list):
            for d, v in zip(df, data_variant.subvariant_names()):
                self._prepare_data_analysis( d, data_variant, v)
            return
        
        self._prepare_data_analysis(df, data_variant)
        
    def _framenums_from_index(self, indexes: np.ndarray, data: np.ndarray) -> List[int]:
        return list(data[indexes, 0])

    def _prepare_data_analysis(self, df: pd.DataFrame, data_variant: DataVariant, subvariant: str = ''):
        data = df.to_numpy()
        rdata = data[:, 1:]
        
        mins = np.round(np.min(rdata, axis=0).astype(np.float64), decimals=2)
        min_fns = self._framenums_from_index(np.argmin(rdata, axis=0), data)
        
        maxs = np.round(np.max(rdata, axis=0).astype(np.float64), decimals=2)
        max_fns = self._framenums_from_index(np.argmax(rdata, axis=0), data)
        
        dv_str = data_variant.shortname()
        dir = f'{self.kitti_locations.analysis_dir}/{dv_str}'
        dir = dir if not subvariant else f'{dir}/{subvariant}'
        os.makedirs(dir, exist_ok=True)
        
        for min_fn, max_fn in zip(min_fns, max_fns):
            self._visualize_frames(data_variant=data_variant, kitti_locations=self.kitti_locations, frame_numbers=[min_fn, max_fn])
            
        stats = np.vstack((mins, min_fns, maxs, max_fns))
        columns = list(map(lambda c: c.capitalize(), list(df.columns)[1:]))

        df = pd.DataFrame(stats, columns=columns)
        df.insert(0, "Name", pd.Series(["Min", "Min Frame Number", "Max", "Max Frame Number"]))

        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        
        filename = f'{dir}/{dv_str}-{now}.csv'
        
        df.to_csv(filename, index=False)

        logging.info(f'Analysis data written to file:///{filename}')
        
    
    def _visualize_frames(self, data_variant: DataVariant, kitti_locations: KittiLocations, frame_numbers: List[str]):
        for frame_number in frame_numbers:
            loader = FrameDataLoader(kitti_locations=kitti_locations, frame_number=frame_number)
            dv_str = data_variant.shortname()
            
            if data_variant in DataVariant.semantic_variants():
                vis2d = Visualization2D(frame_data_loader=loader, classes_visualized=get_class_names())
                vis2d.draw_plot(plot_figure=False, save_figure=True, show_gt=True,
                        show_lidar=True, show_radar=True, subdir=f'analysis/{dv_str}', filename='annotated')
                
                vis2d.draw_plot(plot_figure=False, save_figure=True, show_gt=True,
                        show_lidar=False, show_radar=True, subdir=f'analysis/{dv_str}', filename='radar-annotated')

            imsave(f'{kitti_locations.analysis_dir}/{dv_str}/{frame_number}.png', loader.image)


def prepare_data_analysis(data_manager: DataManager):
    analysis = DataAnalysisHelper(data_manager)
    
    for dv in DataVariant.all_variants():
        analysis.prepare_data_analysis(dv, DataView.BASIC_ANALYSIS)
        #analysis.prepare_data_analysis(dv, DataView.ANALYSIS)