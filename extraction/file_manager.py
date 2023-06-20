import logging
import os
import pandas as pd

from datetime import datetime
from typing import Dict, List, Optional, Union
from extraction.extract import ParameterRangeExtractor
from extraction.helpers import DataVariant
from vod.configuration.file_locations import KittiLocations

class DataManager:
    
    def __init__(self, kitti_locations: KittiLocations) -> None:
        self.kitti_locations = kitti_locations
        self.extractor = ParameterRangeExtractor(kitti_locations)
        self.data: Dict[DataVariant, pd.DataFrame] = {}
        
    def get_df_plot_ready(self, data_variant: DataVariant, refresh=False) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Gets the dataframe for the given data variant either by loading it from an HDF-5 file or by extracting it from the dataset.
        Additionally, modifies the data, so it can be directly used for plotting.

        :param data_variant: the data variant for which the dataframe is to be retrieved

        Returns the dataframe containing the data requested through the data variant
        """
        df = self.get_df(data_variant, refresh)
        
        if data_variant in [DataVariant.SEMANTIC_RAD, DataVariant.SYNTACTIC_RAD]:
            return df.drop([0]) # frame number

        elif data_variant == DataVariant.SEMANTIC_OBJECT_DATA:
            return df.drop([0, 1]) # frame number, class
        
        elif data_variant == DataVariant.SEMANTIC_OBJECT_DATA_BY_CLASS:
            df = [d.drop([0, 1]) for d in df]
            
        elif data_variant == DataVariant.SYNTACTIC_RAD:
            df = [d.drop([0]) for d in df]
        
    def get_df(self, data_variant: DataVariant, refresh=False) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Gets the dataframe for the given data variant either by loading it from an HDF-5 file or by extracting it from the dataset

        :param data_variant: the data variant for which the dataframe is to be retrieved

        Returns the dataframe containing the data requested through the data variant
        """
        if not refresh and (self.data.get(data_variant) is not None or self.load_dataframe(data_variant) is not None):
            return self.data[data_variant]

        if data_variant == DataVariant.SYNTACTIC_RAD:
            self.store_dataframe(
                data_variant, self.extractor.extract_rad_from_syntactic_data())

        elif data_variant == DataVariant.SEMANTIC_RAD:
            object_df: pd.DataFrame = self.get_df(
                data_variant=DataVariant.SEMANTIC_OBJECT_DATA)
            object_df = object_df[DataVariant.SEMANTIC_RAD.column_names()]
            self.data[data_variant] = object_df

        elif data_variant == DataVariant.SEMANTIC_OBJECT_DATA:
            self.store_dataframe(
                data_variant, self.extractor.extract_object_data_from_semantic_data())

        elif data_variant == DataVariant.SEMANTIC_OBJECT_DATA_BY_CLASS:
            object_df = self.get_df(DataVariant.SEMANTIC_OBJECT_DATA)
            object_data_by_class = self.extractor.split_by_class(object_df)
            self.data[data_variant] = object_data_by_class

        elif data_variant == DataVariant.STATIC_DYNAMIC_RAD:
            stat_dyn_rad = self.extractor.split_rad_by_threshold(
                self.get_df(DataVariant.SYNTACTIC_RAD))
            self.data[data_variant] = stat_dyn_rad

        return self.data[data_variant]
    
    def load_dataframe(self, data_variant: DataVariant) -> Optional[pd.DataFrame]:
        """
        Loads a dataframe from the most recently saved HDF5-file for this data variant.

        :param data_variant: the data variant of the dataframe to be loaded
        """
        dv_str = data_variant.name.lower()
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

        dv_str = data_variant.name.lower()
        data_dir = f'{self.kitti_locations.data_dir}'
        os.makedirs(data_dir, exist_ok=True)

        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.data[data_variant] = df
        path = f'{data_dir}/{dv_str}-{now}.hdf5'
        df.to_hdf(path, key=dv_str, mode='w')
        logging.info(f'Data saved in file:///{path}.hdf5')
            