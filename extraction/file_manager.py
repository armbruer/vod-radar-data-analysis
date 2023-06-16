from datetime import datetime
import logging
import os
from typing import Dict, List, Optional
import numpy as np
from extraction.extract import DataVariant

from vod.configuration.file_locations import KittiLocations

class DataManager:
    
    def __init__(self, kitti_locations: KittiLocations) -> None:
        self.kitti_locations = kitti_locations

        self.data: Dict[DataVariant, List[np.ndarray]] = {}
    
    def load_data(self, data_variant: DataVariant) -> Optional[List[np.ndarray]]:
        """
        Loads a data array of shape from the most recently saved numpy file given this data_variant.

        :param data_variant: the data variant of the file to be loaded  
        """
        dv_str = data_variant.name.lower()
        data_dir = f'{self.kitti_locations.data_dir}/{dv_str}'
        os.makedirs(data_dir, exist_ok=True)

        matching_files = []
        for file in os.listdir(data_dir):
            if file.endswith('.npy') and dv_str in file:
                datetime_str = file.split('-')[-1].split('.')[0]
                matching_files.append((file, datetime_str))

        matching_files = sorted(
            matching_files, key=lambda x: datetime.strptime(x[1], '%Y_%m_%d_%H_%M_%S'))

        if not matching_files:
            return None

        most_recent: str = matching_files[-1][0]
        parts = most_recent.split('-')
        data = []
        if len(parts) > 2 and parts[1].isdecimal():
            # we need to load all files in the list now
            i = 0
            while True:
                try:
                    name = f'{parts[0]}-{i}-{parts[2]}'
                    i += 1
                    data.append(np.load(f'{data_dir}/{name}'))
                except:
                    break
        else:
            data = [np.load(f'{data_dir}/{most_recent}')]

        self._data[data_variant] = data
        return data

    def store_data(self, data_variant: DataVariant, data: List[np.ndarray]):
        """
        Stores the data array in a numpy file using data variant in the name of the file.

        :param data_variant: the data_variant of this rad array
        :param data: the data array to be stored
        """
        if not isinstance(data, list):
            raise ValueError('data must be of type list')

        dv_str = data_variant.name.lower()
        data_dir = f'{self.kitti_locations.data_dir}/{dv_str}'
        os.makedirs(data_dir, exist_ok=True)

        now = self._now()
        self._data[data_variant] = data
        for i, d in enumerate(data):
            path = f'{data_dir}/{dv_str}-{i}-{now}.npy'
            np.save(path, d)
            logging.info(f'Data saved in file:///{path}.npy')