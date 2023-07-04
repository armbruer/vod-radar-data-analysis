from itertools import repeat
import logging
import multiprocessing
import os
import numpy as np
import pandas as pd

from typing import List
from extraction.file_manager import DataManager, DataView
from extraction.helpers import DataVariant, DataViewType
from extraction.visualization import visualize_frames

class DataAnalysisHelper:
    
    def __init__(self, data_manager: DataManager) -> None:
        self.data_manager = data_manager
        self.kitti_locations = data_manager.kitti_locations
        
    def prepare_data_analysis(self, data_variant: DataVariant, data_view: DataViewType):
        data_view: DataView = self.data_manager.get_view(data_variant=data_variant, data_view_type=data_view)
        df = data_view.df
        if isinstance(df, list):
            iter = zip(df, repeat(data_variant), data_variant.subvariant_names())
            cpus = int(multiprocessing.cpu_count() * 0.75)
            pool = multiprocessing.Pool(processes=cpus)
            pool.starmap(self._prepare_data_analysis, iter)
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
            visualize_frames(data_variant=data_variant, kitti_locations=self.kitti_locations, frame_numbers=[min_fn, max_fn], locs=[min_loc, max_loc])
            
        stats = np.vstack((mins, min_fns, maxs, max_fns))
        columns = list(map(lambda c: c.capitalize(), list(df.columns)[1:-3]))

        df = pd.DataFrame(stats, columns=columns)
        df.insert(0, "Name", pd.Series(["Min", "Min Frame Number", "Max", "Max Frame Number"]))
        
        filename = f'{dir}/{dv_str}.csv'
        
        df.to_csv(filename, index=False)

        logging.info(f'Analysis data written to file:///{filename}')


def prepare_data_analysis(data_manager: DataManager):
    analysis = DataAnalysisHelper(data_manager)
    
    for dv in DataVariant.all_variants():
        analysis.prepare_data_analysis(dv, DataViewType.BASIC_ANALYSIS)
        #analysis.prepare_data_analysis(dv, DataViewType.ANALYSIS)