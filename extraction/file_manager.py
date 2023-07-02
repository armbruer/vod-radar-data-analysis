import logging
import os
import pandas as pd

from datetime import datetime
from typing import Dict, List, Optional, Union
from extraction.extract import ParameterRangeExtractor
from extraction.helpers import DataVariant, DataViewType
from vod.configuration.file_locations import KittiLocations

class DataView:
    
    def __init__(self, 
                 df: Union[pd.DataFrame, List[pd.DataFrame]], 
                 data_variant: DataVariant, 
                 data_view_type: DataViewType) -> None:
        self.df = df
        self.variant = data_variant
        self.view = data_view_type
        
        self._init_view()
        
    def _init_view(self):
        self._tmp_df: List[pd.DataFrame] = self.df
        if not isinstance(self._tmp_df, list):
            self._tmp_df = [self.df]
            
        self._tmp_df = [df.drop(self.view.columns_to_drop(), axis=1, errors='ignore') for df in self._tmp_df]
            
        drop_indexes = self._tmp_df[0].columns.get_loc(self.view.columns_to_drop())
        
        self.lims = [lim for i, lim in enumerate(self.variant.lims()) if i not in drop_indexes]
        
        self.df = self._tmp_df[0] if len(self._tmp_df) == 1 else self._tmp_df
        
        

class DataManager:
        
    def __init__(self, kitti_locations: KittiLocations) -> None:
        self.kitti_locations = kitti_locations
        self.extractor = ParameterRangeExtractor(kitti_locations)
        self.data: Dict[DataVariant, pd.DataFrame] = {}
        
    def get_view(self, data_variant: DataVariant, data_view_type: DataViewType = DataViewType.NONE, refresh=False) -> DataView:
        """
        Gets the dataframe for the given data variant either by loading it from an HDF-5 file or by extracting it from the dataset.

        :param data_variant: the data variant for which the dataframe is to be retrieved
        :param data_view: the data view to apply to the data variant, this removes columns unneeded in the current context

        Returns the dataframe containing the data requested
        """
        df = self._get_df(data_variant, refresh)
        
        return DataView(df=df, data_variant=data_variant, data_view_type=data_view_type)
        
    def _get_df(self, data_variant: DataVariant, refresh=False) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Gets the dataframe for the given data variant either by loading it from an HDF-5 file or by extracting it from the dataset

        :param data_variant: the data variant for which the dataframe is to be retrieved

        Returns the dataframe containing the data requested through the data variant
        """
        if not refresh and (self.data.get(data_variant) is not None or self.load_dataframe(data_variant) is not None):
            return self.data[data_variant]

        if data_variant == DataVariant.SYNTACTIC_DATA:
            self.store_dataframe(
                data_variant, self.extractor.extract_data_from_syntactic_data())

        elif data_variant == DataVariant.SEMANTIC_DATA:
            self.store_dataframe(
                data_variant, self.extractor.extract_object_data_from_semantic_data())

        elif data_variant == DataVariant.SEMANTIC_DATA_BY_CLASS:
            semantic_df = self._get_df(DataVariant.SEMANTIC_DATA)
            semantic_by_class = self.extractor.split_by_class(semantic_df)
            self.data[data_variant] = semantic_by_class

        elif data_variant == DataVariant.SYNTACTIC_DATA_BY_MOVING:
            syntactic_df = self._get_df(DataVariant.SYNTACTIC_DATA)
            syntactic_by_moving = self.extractor.split_rad_by_threshold(syntactic_df)
            self.data[data_variant] = syntactic_by_moving

        return self.data[data_variant]
    
    def load_dataframe(self, data_variant: DataVariant) -> Optional[pd.DataFrame]:
        """
        Loads a dataframe from the most recently saved HDF5-file for this data variant.

        :param data_variant: the data variant of the dataframe to be loaded
        """
        dv_str = data_variant.shortname()
        data_dir = f'{self.kitti_locations.data_dir}'
        os.makedirs(data_dir, exist_ok=True)

        matching_files = []
        for file in os.listdir(data_dir):
            if file.endswith('.hdf5') and dv_str in file:
                datetime_str = file.split('-')[-1].split('.')[0]
                matching_files.append((file, datetime_str))

        matching_files = sorted(
            matching_files, key=lambda x: datetime.strptime(x[1], '%Y_%m_%d_%H_%M_%S'))

        if not matching_files:
            return None

        most_recent: str = matching_files[-1][0]
        
        df = pd.read_hdf(f'{data_dir}/{most_recent}', key=dv_str)

        self.data[data_variant] = df
        return df

    def store_dataframe(self, data_variant: DataVariant, df: pd.DataFrame):
        """
        Stores the dataframe in an HDF-5 file using the data variant in the file path.

        :param data_variant: the data_variant of the data to be stored
        :param data: the dataframe to be stored
        """
        if isinstance(df, list):
            raise ValueError('df must not be of type list')

        dv_str = data_variant.shortname()
        data_dir = f'{self.kitti_locations.data_dir}'
        os.makedirs(data_dir, exist_ok=True)

        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.data[data_variant] = df
        path = f'{data_dir}/{dv_str}-{now}.hdf5'
        df.to_hdf(path, key=dv_str, mode='w')
        logging.info(f'Data saved in file:///{path}')
            