# TODO
import sys
import os
sys.path.append(os.path.abspath("../view-of-delft-dataset"))

from typing import Dict, List
from vod.common.file_handling import get_frame_list_from_folder
from vod.frame import FrameDataLoader
from vod.frame import FrameTransformMatrix
from vod.configuration.file_locations import KittiLocations
import extraction as ex
from tqdm import tqdm
from datetime import datetime

import numpy as np
import os
from enum import Enum

class DataVariant(Enum):
    SYNTACTIC_RAD = 0,
    SEMANTIC_RAD = 1,
    STATIC_RAD = 2,
    DYNAMIC_RAD = 3,
    SEMANTIC_OBJECT_DATA = 4

class ParameterRangeExtractor:
    
    def __init__(self, kitti_locations: KittiLocations) -> None:
        self.kitti_locations = kitti_locations
        
        self._data: Dict[DataVariant, np.ndarray] = {}
    
    def get_data(self, data_variant: DataVariant) -> np.ndarray:
        """
        Gets the array data for the given data variant either directly from file or by extracting it from the respective frames.
        
        :param data_variant: the data variant for which the data array is to be retrieved
        
        Returns the array containing the data requested in data_variant
        """
        if self._data.get(data_variant) is not None:
            return self._data[data_variant]
        
        try:
            self._data[data_variant] = self._load_data(data_variant)
        except FileNotFoundError as e:
            if not "No matching data file found'" in str(e):
                raise e
            
            if data_variant == DataVariant.SYNTACTIC_RAD:
                self._store_data(data_variant, self._extract_rad_from_syntactic_data())
            elif data_variant == DataVariant.SEMANTIC_RAD:
                object_data = self.get_data(DataVariant.SEMANTIC_OBJECT_DATA)
                self._store_data(data_variant, object_data[:, 4:])
            elif data_variant == DataVariant.SEMANTIC_OBJECT_DATA:
                self._store_data(data_variant, self._extract_object_data_from_semantic_data())
            elif data_variant == DataVariant.DYNAMIC_RAD or data_variant == DataVariant.STATIC_RAD:
                static, dynamic = self._split_rad(self.get_data(DataVariant.SYNTACTIC_RAD))
                self._store_data(DataVariant.STATIC_RAD, static)
                self._store_data(DataVariant.DYNAMIC_RAD, dynamic)
            
                
        return self._data[data_variant]
    
    @staticmethod
    def names_rad():
        return ["range", "azimuth", "doppler"]
    
    
    @staticmethod
    def names_rad_with_unit():
        return ["range (m)", "azimuth (degree)", "doppler (m/s)"]
    
    
    @staticmethod
    def names_object_data():
        return ["class", "velocity", "detections", "bbox volume", "range", "azimuth", "doppler"]
    
    
    @staticmethod
    def names_object_data_with_unit():
        return ["class", "velocity (m/s)", "detections (#)", "bbox volume (m^3)", "range (m)", "azimuth (degree)", "doppler (m/s)"]
    
    
    def _extract_rad_from_syntactic_data(self) -> np.ndarray:
        """
        Extract the range, azimuth, doppler values for each frame and detection in this dataset.
        This method works on the syntactic (unannotated) data of the dataset.
        
        
        Returns a numpy array of shape (-1, 3) with columns range, azimuth, doppler.
        """
        frame_numbers = get_frame_list_from_folder(self.kitti_locations.radar_dir, labels=False)
        
        ranges: List[np.ndarray] = []
        azimuths: List[np.ndarray] = []
        dopplers: List[np.ndarray] = []
        
        # TODO: Optimally one would like to split this into multiple parts to use less memory at once...
        for frame_number in tqdm(iterable=frame_numbers, desc='Syntactic RAD: Going through frames'):
            loader = FrameDataLoader(kitti_locations=self.kitti_locations, frame_number=frame_number)
            
            # radar_data shape: [x, y, z, RCS, v_r, v_r_compensated, time] (-1, 7)
            radar_data = loader.radar_data 
            
            ranges.append(ex.locs_to_distance(radar_data[:, :3]))
            azimuths.append(np.rad2deg(ex.azimuth_angle_from_location(radar_data[:, :2])))
            dopplers.append(radar_data[:, 4])
        
        return np.array([ranges, azimuths, dopplers]).T
    


    def _split_rad(self, rad: np.ndarray, static_object_doppler_threshold: float = 0.5) -> List[np.ndarray]:
        """
        Splits the RAD array into two arrays according to a doppler threshold value.
        
        
        :param rad: the rad array to be split
        :param static_object_doppler_threshold: the threshold value to split the arrays into two arrays 
        
        Returns two RAD arrays, the first resulting array contains static detections and the second contains dynamic detections
        """
        cond = np.abs(rad[:, 2]) < static_object_doppler_threshold
        
        return rad[cond], rad[~cond]
    

    def _extract_object_data_from_semantic_data(self) -> np.ndarray:
        """
        For each object in the frame retrieve the following data: object tracking id, object class, absolute velocity, 
        number of detections, bounding box volume, ranges, azimuths, relative velocity (doppler).
        
        Returns a numpy array of shape (-1, 7) with columns range, azimuth, doppler.
        """
        
        # only those frame_numbers which have annotations
        frame_numbers = get_frame_list_from_folder(self.kitti_locations.label_dir)
        
        object_data_list: List[np.ndarray] = []
        
        # TODO: Optimally one would like to split this into multiple parts to use less memory at once...
        for frame_number in tqdm(iterable=frame_numbers, desc='Semantic data: Going through frames'):
            loader = FrameDataLoader(kitti_locations=self.kitti_locations, frame_number=frame_number)
            transforms = FrameTransformMatrix(frame_data_loader_object=loader)
            
            object_data: np.ndarray = ex.get_data_for_objects_in_frame(loader=loader, transforms=transforms)
            object_data_list.append(object_data)
        
        
        return np.vstack(object_data_list)
    
    def _now(self): return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    def _load_data(self, data_variant: DataVariant) -> np.ndarray:
        """
        Loads a data array of shape from the most recently saved numpy file given this data_variant.
        
        :param data_variant: the data variant of the file to be loaded  
        """
        data_variant = data_variant.name.lower()
        matching_files = []
        for file in os.listdir(self.kitti_locations.output_dir):
            if file.endswith('.npy') and data_variant in file:
                datetime_str = file.split('-')[1].split('.')[0]
                matching_files.append((file, datetime_str))
        
        matching_files = sorted(matching_files, key=lambda x: datetime.strptime(x[1], '%Y_%m_%d_%H_%M_%S'))
        
        if not matching_files:
            raise FileNotFoundError('No matching data file found')
        
        most_recent = matching_files[-1][0]
        data = np.load(f'{self.kitti_locations.output_dir}/{most_recent}')
        
        return data
    
    
    def _store_data(self, data_variant: DataVariant, data: np.ndarray):
        """
        Stores the data array in a numpy file using data variant in the name of the file.
        
        :param data_variant: the data_variant of this rad array
        :param data: the data array to be stored
        """
        
        self._data[data_variant] = data
        np.save(f'{self.kitti_locations.output_dir}/{data_variant.name.lower()}-{self._now()}.npy', data)
