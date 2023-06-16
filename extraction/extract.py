import numpy as np
import extraction as ex

from extraction.file_manager import DataManager
from enum import Enum
from datetime import datetime
from tqdm import tqdm
from vod.frame import FrameTransformMatrix
from vod.frame import FrameDataLoader
from vod.common.file_handling import get_frame_list_from_folder
from typing import List
from collections import defaultdict


class DataVariant(Enum):
    SYNTACTIC_RAD = 0,
    SEMANTIC_RAD = 1,
    STATIC_DYNAMIC_RAD = 2,
    SEMANTIC_OBJECT_DATA = 3,
    SEMANTIC_OBJECT_DATA_BY_CLASS = 4

    def column_names(self, with_unit: bool = False) -> List[str]:
        if self == DataVariant.SEMANTIC_RAD or self == DataVariant.STATIC_DYNAMIC_RAD or self == DataVariant.SYNTACTIC_RAD:
            if with_unit:
                return ["range (m)", "azimuth (degree)", "doppler (m/s)"]
            else:
                return ["range", "azimuth", "doppler"]
        elif self == DataVariant.SEMANTIC_OBJECT_DATA_BY_CLASS or self == DataVariant.SEMANTIC_OBJECT_DATA:
            # we never want the class, even though it is included
            if with_unit:
                return ["class", "velocity (m/s)", "detections (#)", "bbox volume (m^3)", "range (m)", "azimuth (degree)", "doppler (m/s)"]
            else:
                return ["class", "detections", "bbox volume", "range", "azimuth", "doppler"]

        return []

    def index_to_str(self, index) -> str:
        if self == DataVariant.SEMANTIC_OBJECT_DATA_BY_CLASS:
            return ex.name_from_class_id(index)
        elif self == DataVariant.STATIC_DYNAMIC_RAD:
            if index == 0:
                return "static_rad"
            else:
                return "dynamic_rad"

        return ''


class ParameterRangeExtractor:

    def __init__(self, data_manager: DataManager) -> None:
        self.data_manager = data_manager
        self.data = data_manager.data

    def get_data(self, data_variant: DataVariant, refresh=False) -> List[np.ndarray]:
        """
        Gets the array data for the given data variant either directly from file or by extracting it from the respective frames.

        :param data_variant: the data variant for which the data array is to be retrieved

        Returns the array containing the data requested in data_variant
        """
        if not refresh and (self.data.get(data_variant) is not None or self.data_manager.load_data(data_variant) is not None):
            return self.data[data_variant]

        if data_variant == DataVariant.SYNTACTIC_RAD:
            self.data_manager.store_data(
                data_variant, self._extract_rad_from_syntactic_data())

        elif data_variant == DataVariant.SEMANTIC_RAD:
            object_data: np.ndarray = self.get_data(
                DataVariant.SEMANTIC_OBJECT_DATA)
            self.data_manager.store_data(data_variant, [object_data[0][:, 4:]])

        elif data_variant == DataVariant.SEMANTIC_OBJECT_DATA:
            self.data_manager.store_data(
                data_variant, self._extract_object_data_from_semantic_data())

        elif data_variant == DataVariant.SEMANTIC_OBJECT_DATA_BY_CLASS:
            object_data = self.get_data(DataVariant.SEMANTIC_OBJECT_DATA)
            object_data_by_class = self._split_by_class(object_data[0])
            self.data_manager.store_data(data_variant, object_data_by_class)

        elif data_variant == DataVariant.STATIC_DYNAMIC_RAD:
            stat_dyn_rad = self._split_rad_by_threshold(
                *self.get_data(DataVariant.SYNTACTIC_RAD))
            self.data_manager.store_data(DataVariant.STATIC_DYNAMIC_RAD, stat_dyn_rad)

        return self.data[data_variant]

    def _extract_rad_from_syntactic_data(self) -> List[np.ndarray]:
        """
        Extract the range, azimuth, doppler values for each frame and detection in this dataset.
        This method works on the syntactic (unannotated) data of the dataset.


        Returns a numpy array of shape (-1, 3) with columns range, azimuth, doppler.
        """
        frame_numbers = get_frame_list_from_folder(
            self.kitti_locations.radar_dir, labels=False)

        ranges: List[np.ndarray] = []
        azimuths: List[np.ndarray] = []
        dopplers: List[np.ndarray] = []

        for frame_number in tqdm(iterable=frame_numbers, desc='Syntactic RAD: Going through frames'):
            loader = FrameDataLoader(
                kitti_locations=self.kitti_locations, frame_number=frame_number)

            # radar_data shape: [x, y, z, RCS, v_r, v_r_compensated, time] (-1, 7)
            radar_data = loader.radar_data
            if radar_data is not None:

                ranges.append(ex.locs_to_distance(radar_data[:, :3]))
                azimuths.append(np.rad2deg(
                    ex.azimuth_angle_from_location(radar_data[:, :2])))
                dopplers.append(radar_data[:, 4])

        return [np.vstack(list(map(np.hstack, [ranges, azimuths, dopplers]))).T]

    def _split_by_class(self, object_data: np.ndarray) -> List[np.ndarray]:
        """
        Splits the object_data by class into a new array with a third axis for the class.
        """

        by_class = defaultdict(list)

        for i in range(object_data.shape[0]):
            class_id = object_data[i, 0]
            by_class[class_id].append(object_data[i])

        by_class = dict(sorted(by_class.items()))

        return [np.vstack(v) for _, v in by_class.items()]

    def _split_rad_by_threshold(self, rad: np.ndarray, static_object_doppler_threshold: float = 0.5) -> List[np.ndarray]:
        """
        Splits the RAD by a threshold value for static object into two parts by adding a third dimension to the array. 
        The static objects are index 0, the dynamic objects are index 1


        :param rad: the rad array to be split
        :param static_object_doppler_threshold: the threshold value to split the arrays into two arrays 

        Returns an array with a third dimension depending on whether the detections are static (index 0) or dynamic (index 1)
        """
        cond = np.abs(rad[:, 2]) < static_object_doppler_threshold

        return [rad[cond], rad[~cond]]

    def _extract_object_data_from_semantic_data(self) -> List[np.ndarray]:
        """
        For each object in the frame retrieve the following data: object tracking id, object class, absolute velocity, 
        number of detections, bounding box volume, ranges, azimuths, relative velocity (doppler).

        Returns a numpy array of shape (-1, 7) with columns range, azimuth, doppler.
        """

        # only those frame_numbers which have annotations
        frame_numbers = get_frame_list_from_folder(
            self.kitti_locations.label_dir)

        object_data_list: List[np.ndarray] = []

        for frame_number in tqdm(iterable=frame_numbers, desc='Semantic data: Going through frames'):
            loader = FrameDataLoader(
                kitti_locations=self.kitti_locations, frame_number=frame_number)
            transforms = FrameTransformMatrix(frame_data_loader_object=loader)

            object_data = ex.get_data_for_objects_in_frame(
                loader=loader, transforms=transforms)
            if object_data is not None:
                object_data_list.append(object_data)

        return [np.vstack(object_data_list)]

    def _now(self): return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
