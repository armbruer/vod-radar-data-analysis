# TODO
import sys
import os
sys.path.append(os.path.abspath("../view-of-delft-dataset"))

from typing import Dict, List
from vod.common.file_handling import get_frame_list_from_folder
from vod.evaluation.evaluation_common import get_label_annotations
from vod.frame import FrameDataLoader
from vod.frame import FrameTransformMatrix
from vod.configuration.file_locations import KittiLocations
from vod.frame import homogenous_transformation_cartesian_coordinates
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
    DYNAMIC_RAD = 3
    
    @staticmethod
    def column_names():
        return ["range", "azimuth", "doppler"]
    
    @staticmethod
    def column_names_with_unit():
        return ["range (m)", "azimuth (degree)", "doppler (m/s)"]

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
            self._data[data_variant] = self._load_rad(data_variant)
        except FileNotFoundError as e:
            if not "rad" in str(e):
                raise e
            
            if data_variant == DataVariant.SYNTACTIC_RAD:
                self._store_rad(data_variant, self._extract_rad_from_syntactic_data())
            elif data_variant == DataVariant.SEMANTIC_RAD:
                self._store_rad(data_variant, self._extract_rad_from_semantic_data())
            elif data_variant == DataVariant.DYNAMIC_RAD or data_variant == DataVariant.STATIC_RAD:
                static, dynamic = self._split_rad(self.get_data(DataVariant.SYNTACTIC_RAD))
                self._store_rad(DataVariant.STATIC_RAD, static)
                self._store_rad(DataVariant.DYNAMIC_RAD, dynamic)
                
        return self._data[data_variant]
    
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
        
        res = np.vstack(list(map(np.hstack, [ranges, azimuths, dopplers]))).T
        return res



    def _split_rad(self, rad: np.ndarray, static_object_doppler_threshold: float = 0.5) -> List[np.ndarray]:
        """
        Splits the RAD array into two arrays according to a doppler threshold value.
        
        
        :param rad: the rad array to be split
        :param static_object_doppler_threshold: the threshold value to split the arrays into two arrays 
        
        Returns two RAD arrays, the first resulting array contains static detections and the second contains dynamic detections
        """
        cond = np.abs(rad[:, 2]) < static_object_doppler_threshold
        
        return rad[cond], rad[~cond]
        
        

    def _extract_rad_from_semantic_data(self) -> np.ndarray:
        """
        Get the Range, Azimuth, Doppler values for each frame and object in this dataset.
        This method works on the semantic (annotated) data of the dataset.
        
        Returns a numpy array of shape (-1, 3) with columns range, azimuth, doppler.
        """
        
        annotations = get_label_annotations(self.kitti_locations.label_dir)
        locations: List[np.ndarray] = []
        azimuths: List[np.ndarray] = []
        dopplers: List[np.ndarray] = []

        for anno in tqdm(iterable=annotations, desc='Semantic RAD: Going through annotations'):
            # operating here on a per frame basis!#
            frame_number = anno['frame_number']
            location_values: np.ndarray = anno['location'] # (-1, 3)
            azimuth_values: np.ndarray = anno['alpha']
            
            loader = FrameDataLoader(kitti_locations=self.kitti_locations, frame_number=frame_number)
            transforms = FrameTransformMatrix(frame_data_loader_object=loader)
            
            # Transform locations to cartesian coordinates
            locations_transformed = homogenous_transformation_cartesian_coordinates(location_values, transforms.t_camera_radar)
            locations.append(locations_transformed)

            azimuths.append(azimuth_values)
            
            doppler_values: np.ndarray = ex.dopplers_for_objects_in_frame(loader=loader, transforms=transforms)
            dopplers.append(doppler_values)
        
        
        locations, azimuths, dopplers = map(np.hstack, [locations, azimuths, dopplers])
        azimuths = np.rad2deg(azimuths)
        ranges = ex.locs_to_distance(locations)
        
        return np.vstack((ranges, azimuths, dopplers)).T
    
    def _now(self): return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    def _load_rad(self, data_variant: DataVariant) -> np.ndarray:
        """
        Loads a RAD array of shape (-1, 3) from the most recently saved numpy file given this data_variant.
        
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
            raise FileNotFoundError('No matching rad file found')
        
        most_recent = matching_files[-1][0]
        rad = np.load(f'{self.kitti_locations.output_dir}/{most_recent}')
        
        return rad
    
    
    def _store_rad(self, data_variant: DataVariant, rad: np.ndarray):
        """
        Stores the rad array in a numpy file using data variant in the name of the file.
        
        :param data_variant: the data_variant of this rad array
        :param rad: the rad array of shape (-1, 3) to be stored
        """
        
        self._data[data_variant] = rad
        np.save(f'{self.kitti_locations.output_dir}/{data_variant.name.lower()}-{self._now()}.npy', rad)
