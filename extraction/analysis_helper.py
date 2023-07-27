from itertools import repeat
import logging
import multiprocessing
import os
import numpy as np
import pandas as pd

from typing import List, Optional
from extraction.file_manager import DataManager, DataView
from extraction.helpers import DataVariant, DataViewType
from extraction.visualization import visualize_frame

class DataAnalysisHelper:
    
    runs_counter = 0
    
    def __init__(self, data_manager: DataManager) -> None:
        self.data_manager = data_manager
        self.kitti_locations = data_manager.kitti_locations
    
    def prepare_data_analysis(self, 
                              data_variant: DataVariant, 
                              data_view_type: DataViewType, 

                              frame_numbers: Optional[List[str]]=None):
        
        DataAnalysisHelper.runs_counter += 1 # make sure they are all unique
        data_view: DataView = self.data_manager.get_view(data_variant=data_variant, 
                                                         data_view_type=data_view_type, frame_numbers=frame_numbers)
        df = data_view.df
        if isinstance(df, list):
            iter = zip(df, repeat(data_variant), data_variant.subvariant_names())
            cpus = int(multiprocessing.cpu_count() * 0.75)
            try:
                pool = multiprocessing.Pool(processes=cpus)
                pool.starmap(self._prepare_data_analysis, iter)
            finally:
                pool.close()
                pool.join()
            
            # alternatively without multiprocessing:
            
            # for d, s in zip(df, data_variant.subvariant_names()):
            #     self._prepare_data_analysis(d, data_variant, s)
            return
        
        self._prepare_data_analysis(df, data_variant)

    def _prepare_data_analysis(self, 
                               df: pd.DataFrame, 
                               data_variant: DataVariant, 
                               subvariant: str = ''):
        
        stats_only_view: DataView = DataView(df, data_variant, DataViewType.MIN_MAX_USEFUL)
        stats_only_df: pd.DataFrame = stats_only_view.df
        
        data = df.to_numpy()
        stats_only_data = stats_only_df.to_numpy()

        # 0. Collect data        
        mins = np.round(np.min(stats_only_data, axis=0).astype(np.float64), decimals=2)
        maxs = np.round(np.max(stats_only_data, axis=0).astype(np.float64), decimals=2)
        min_indexes = np.argmin(stats_only_data, axis=0)
        max_indexes = np.argmax(stats_only_data, axis=0)
        min_fns = data[min_indexes, 0]
        max_fns = data[max_indexes, 0]
        
        min_max_indexes = list(min_indexes) + list(max_indexes)
        
        min_max_rows = data[min_max_indexes]
        
        # 1. Create output dir
        dv_str = data_variant.shortname()
        dir = self._create_output_dir(dv_str, subvariant)
        
        # 2. Visualize each frame
        for i, extremum in enumerate(list(min_max_rows)):
            frame_number = extremum[0]
            center_radar = extremum[-3:] # x, y, z
            detections = extremum[7]
            
            visualize_frame(data_variant=data_variant, 
                             kitti_locations=self.kitti_locations, 
                             frame_number=frame_number, 
                             center_radar=center_radar,
                             detections=detections,
                             i=i,
                             runs_counter=DataAnalysisHelper.runs_counter)
            
        stats = np.vstack((mins, min_fns, maxs, max_fns))

        df_res_stats = pd.DataFrame(stats, columns=stats_only_df.columns)
        df_res_stats.insert(0, "Name", pd.Series(["Min", "Min Frame Number", "Max", "Max Frame Number"]))
        
        filename = f'{dir}/{dv_str}-{DataAnalysisHelper.runs_counter}.csv'
        df_res_stats.to_csv(filename, index=False)
        
        
        df_full = pd.DataFrame(data=min_max_rows, columns=df.columns)
        df_full = df_full.round(decimals=2)
        
        filename = f'{dir}/full-data-{dv_str}-{DataAnalysisHelper.runs_counter}'
        df_full.to_csv(f'{filename}.csv', index=False)
        df_full.to_latex(
            f'{filename}.tex',
            float_format="%.2f",
            label=f"table:{filename}",
            position="htb!",
            column_format=len(df_full.columns) * "c",
            index=False,
        )
        logging.info(f'Analysis data written to file:///{filename}')

    def _create_output_dir(self, dv_str, subvariant):
        dir = f'{self.kitti_locations.analysis_dir}/{dv_str}'
        dir = dir if not subvariant else f'{dir}/{subvariant}'
        os.makedirs(dir, exist_ok=True)
        return dir


def prepare_data_analysis(data_manager: DataManager):
    analysis = DataAnalysisHelper(data_manager)
    
    for dv in DataVariant.all_variants():
        analysis.prepare_data_analysis(dv, DataViewType.NONE)