import numpy as np
import pandas as pd

from tqdm import tqdm
from extraction.helpers import azimuth_angle_from_location, get_data_for_objects_in_frame, locs_to_distance, DataVariant
from vod.configuration.file_locations import KittiLocations
from vod.frame import FrameTransformMatrix
from vod.frame import FrameDataLoader
from vod.common.file_handling import get_frame_list_from_folder
from typing import Dict, List


class ParameterRangeExtractor:

    def __init__(self, kitti_locations: KittiLocations) -> None:
        self.kitti_locations = kitti_locations

    def extract_data_from_syntactic_data(self) -> pd.DataFrame:
        """
        Extract the frame number, range, azimuth, doppler values and loactions for each detection in this dataset.
        This method works on the syntactic (unannotated) data of the dataset.


        Returns a dataframe with columns frame_number, range, azimuth, doppler, x, y, z
        """
        frame_numbers = get_frame_list_from_folder(
            self.kitti_locations.radar_dir, fileending='.bin')

        ranges: List[np.ndarray] = []
        azimuths: List[np.ndarray] = []
        dopplers: List[np.ndarray] = []
        frame_nums: List[np.ndarray] = []
        x: List[np.ndarray] = []
        y: List[np.ndarray] = []
        z: List[np.ndarray] = []
        
        # TODO future work: rcs and v_r_compensated?
        for frame_number in tqdm(iterable=frame_numbers, desc='Syntactic data: Going through frames'):
            loader = FrameDataLoader(
                kitti_locations=self.kitti_locations, frame_number=frame_number)

            # radar_data shape: [x, y, z, RCS, v_r, v_r_compensated, time] (-1, 7)
            radar_data = loader.radar_data
            if radar_data is not None:

                frame_nums.append(np.full(radar_data.shape[0], frame_number))
                ranges.append(locs_to_distance(radar_data[:, :3]))
                azimuths.append(azimuth_angle_from_location(radar_data[:, :2]))
                dopplers.append(radar_data[:, 4])
                
                # IMPORTANT: see docs/figures/Prius_sensor_setup_5.png (radar) for the directions of these variables
                x.append(radar_data[:, 0])
                y.append(radar_data[:, 1])
                z.append(radar_data[:, 2])

        rad = list(map(np.hstack, [frame_nums, ranges, azimuths, dopplers, x, y, z]))
        cols = DataVariant.SYNTACTIC_DATA.column_names()
        # we construct via series to keep the datatype correct
        return pd.DataFrame({ name : pd.Series(content) for name, content in zip(cols, rad)})

    def split_by_class(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Splits the dataframe by class into a list of dataframes.
        The list of dataframes is sorted according to class_id.
        """
        group_by_class_id = {class_id: x for class_id, x in df.groupby(df['Class'])}
        return list(dict(sorted(group_by_class_id.items())).values())

    def split_rad_by_threshold(self, df: pd.DataFrame, static_object_doppler_threshold: float = 0.5) -> List[pd.DataFrame]:
        """
        Splits the dataframe by a threshold value for static object into two parts by adding a third dimension to the array. 
        The static objects are at index 0. The dynamic objects are at index 1.


        :param df: the RAD dataframe to be split
        :param static_object_doppler_threshold: the threshold value to split the dataframe into two

        Returns a list of dataframes, where the first contains static objects only and the second dynamic objects
        """
        mask = df['Doppler [m/s]'].abs() < static_object_doppler_threshold

        return [df[mask], df[~mask]]

    def extract_object_data_from_semantic_data(self) -> pd.DataFrame:
        """
        For each object in the frame retrieve the following data: frame number, object class, absolute velocity, 
        number of detections, bounding box volume, ranges, azimuths, relative velocity (doppler).

        Returns a dataframe shape (-1, 8) with the columns listed above
        """

        # only those frame_numbers which have annotations
        frame_numbers = get_frame_list_from_folder(
            self.kitti_locations.label_dir)

        object_data_dict: Dict[int, List[np.ndarray]] = {}

        for frame_number in tqdm(iterable=frame_numbers, desc='Semantic data: Going through frames'):
            loader = FrameDataLoader(
                kitti_locations=self.kitti_locations, frame_number=frame_number)
            transforms = FrameTransformMatrix(frame_data_loader_object=loader)

            object_data = get_data_for_objects_in_frame(
                loader=loader, transforms=transforms)
            if object_data is not None:
                for i, param in enumerate(object_data):
                    object_data_dict.setdefault(i, []).append(param)
                
      
        object_data = map(np.hstack, object_data_dict.values())
        
        cols = DataVariant.SEMANTIC_DATA.column_names()
        # we construct via series to keep the datatype correct
        return pd.DataFrame({ name : pd.Series(content) for name, content in zip(cols, object_data)})
