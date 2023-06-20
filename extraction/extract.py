import numpy as np
import pandas as pd

from tqdm import tqdm
from extraction.helpers import azimuth_angle_from_location, get_data_for_objects_in_frame, locs_to_distance, DataVariant
from vod.configuration.file_locations import KittiLocations
from vod.frame import FrameTransformMatrix
from vod.frame import FrameDataLoader
from vod.common.file_handling import get_frame_list_from_folder
from typing import List


class ParameterRangeExtractor:

    def __init__(self, kitti_locations: KittiLocations) -> None:
        self.kitti_locations = kitti_locations

    def extract_rad_from_syntactic_data(self) -> pd.DataFrame:
        """
        Extract the range, azimuth, doppler values for each frame and detection in this dataset.
        This method works on the syntactic (unannotated) data of the dataset.


        Returns a dataframe with columns range, azimuth, doppler.
        """
        frame_numbers = get_frame_list_from_folder(
            self.kitti_locations.radar_dir, fileending='.bin')

        ranges: List[np.ndarray] = []
        azimuths: List[np.ndarray] = []
        dopplers: List[np.ndarray] = []

        for frame_number in tqdm(iterable=frame_numbers, desc='Syntactic RAD: Going through frames'):
            loader = FrameDataLoader(
                kitti_locations=self.kitti_locations, frame_number=frame_number)

            # radar_data shape: [x, y, z, RCS, v_r, v_r_compensated, time] (-1, 7)
            radar_data = loader.radar_data
            if radar_data is not None:

                ranges.append(locs_to_distance(radar_data[:, :3]))
                azimuths.append(np.rad2deg(
                    azimuth_angle_from_location(radar_data[:, :2])))
                dopplers.append(radar_data[:, 4])

        rad = np.vstack(list(map(np.hstack, [ranges, azimuths, dopplers]))).T
        return pd.DataFrame(rad, columns=DataVariant.SYNTACTIC_RAD.column_names())

    def split_by_class(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Splits the dataframe by class list of dataframes
        """
        
        return [x for _, x in df.groupby(df['class'])]

    def split_rad_by_threshold(self, df: pd.DataFrame, static_object_doppler_threshold: float = 0.5) -> List[pd.DataFrame]:
        """
        Splits the dataframe by a threshold value for static object into two parts by adding a third dimension to the array. 
        The static objects are at index 0. The dynamic objects are at index 1.


        :param df: the RAD dataframe to be split
        :param static_object_doppler_threshold: the threshold value to split the dataframe into two

        Returns a list of dataframes, where the first contains static objects only and the second dynamic objects
        """
        mask = df['doppler (m/s)'].abs() < static_object_doppler_threshold

        return [df[mask], df[~mask]]

    def extract_object_data_from_semantic_data(self) -> pd.DataFrame:
        """
        For each object in the frame retrieve the following data: object tracking id, object class, absolute velocity, 
        number of detections, bounding box volume, ranges, azimuths, relative velocity (doppler).

        Returns a dataframe shape (-1, 7) with the columns listed above
        """

        # only those frame_numbers which have annotations
        frame_numbers = get_frame_list_from_folder(
            self.kitti_locations.label_dir)

        object_data_list: List[np.ndarray] = []

        for frame_number in tqdm(iterable=frame_numbers, desc='Semantic data: Going through frames'):
            loader = FrameDataLoader(
                kitti_locations=self.kitti_locations, frame_number=frame_number)
            transforms = FrameTransformMatrix(frame_data_loader_object=loader)

            object_data = get_data_for_objects_in_frame(
                loader=loader, transforms=transforms)
            if object_data is not None:
                object_data_list.append(object_data)

        object_data = np.vstack(object_data_list)
        return pd.DataFrame(object_data, columns=DataVariant.SEMANTIC_OBJECT_DATA.column_names())
