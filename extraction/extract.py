import numpy as np
import pandas as pd
import extraction as ex

from tqdm import tqdm
from extraction.helpers import DataVariant, find_matching_points_for_bboxes, prepare_radar_data
from vod.configuration.file_locations import KittiLocations
from vod.frame import FrameTransformMatrix
from vod.frame import FrameDataLoader
from vod.frame.labels import FrameLabels
from vod.frame.transformations import homogenous_transformation_cart
from vod.common.file_handling import get_frame_list_from_folder
from typing import Dict, List, Optional, Tuple


class ParameterRangeExtractor:

    def __init__(self, kitti_locations: KittiLocations) -> None:
        self.kitti_locations = kitti_locations

    def extract_data_from_syntactic_data(self, frame_numbers: Optional[List[str]]=None) -> pd.DataFrame:
        """
        Extract the frame number, range, azimuth, doppler values and loactions for each detection in this dataset.
        This method works on the syntactic (unannotated) data of the dataset.

        Returns a dataframe with columns frame_number, range, azimuth, doppler, relative velocity (doppler) compensated, x, y, z
        """
        if frame_numbers is None:
            frame_numbers = get_frame_list_from_folder(
                self.kitti_locations.radar_dir, fileending='.bin')

        ranges: List[np.ndarray] = []
        azimuths: List[np.ndarray] = []
        dopplers: List[np.ndarray] = []
        elevations: List[np.ndarray] = []
        frame_nums: List[np.ndarray] = []
        dopplers_compensated: List[np.ndarray] = []
        x: List[np.ndarray] = []
        y: List[np.ndarray] = []
        z: List[np.ndarray] = []
        
        for frame_number in tqdm(iterable=frame_numbers, desc='Syntactic data: Going through frames'):
            loader = FrameDataLoader(kitti_locations=self.kitti_locations, frame_number=frame_number)

            # TODO future work: rcs
            radar_data_r = prepare_radar_data(loader)
            
            if radar_data_r is None:
                continue
            
            frame_nums.append(np.full(radar_data_r.shape[0], frame_number))
            ranges.append(ex.locs_to_distance(radar_data_r[:, :3]))
            azimuths.append(ex.azimuth_angle_from_location(radar_data_r[:, :2]))
            elevations.append(ex.elevation_angle_from_location(radar_data_r[:, [0, 2]]))
            dopplers.append(radar_data_r[:, 4])
            dopplers_compensated.append(radar_data_r[:, 5])
            
            x.append(radar_data_r[:, 0])
            y.append(radar_data_r[:, 1])
            z.append(radar_data_r[:, 2])

        columns = [frame_nums, ranges, azimuths, dopplers, elevations, dopplers_compensated, x, y, z]
        data = list(map(np.hstack, columns))
        col_names = DataVariant.SYNTACTIC_DATA.column_names()
        # we construct via series to keep the datatype correct
        return pd.DataFrame({ name : pd.Series(content) for name, content in zip(col_names, data)})

    def extract_object_data_from_semantic_data(self, frame_numbers: Optional[List[str]]=None) -> pd.DataFrame:
        """
        For each object in the frame retrieve the following data: frame number, ranges, azimuths, 
        object class, summarized class, relative velocity compensated, number of detections, 
        bounding box volume, relative velocity (doppler), height, width, length, x, y, z.

        Returns: a pandas dataframe with the following columns: frame number, ranges, azimuths, 
        object class, summarized class, relative velocity compensated, number of detections, 
        bounding box volume, relative velocity (doppler), height, width, length, x, y, z
        """

        if frame_numbers is None:
            # only those frame_numbers which have annotations
            frame_numbers = get_frame_list_from_folder(self.kitti_locations.label_dir)

        object_data_dict: Dict[int, List[np.ndarray]] = {}

        for frame_number in tqdm(iterable=frame_numbers, desc='Semantic data: Going through frames'):
            loader = FrameDataLoader(kitti_locations=self.kitti_locations, frame_number=frame_number)
            
            labels = loader.get_labels()
            if labels is None:
                continue
            
            labels = FrameLabels(labels)
            transforms = FrameTransformMatrix(frame_data_loader_object=loader)

            object_data = self._get_data_for_objects_in_frame(
                loader=loader, transforms=transforms, labels=labels)
            
            if object_data is None:
                continue
            
            for i, param in enumerate(object_data):
                object_data_dict.setdefault(i, []).append(param)
                
      
        object_data = map(np.hstack, object_data_dict.values())
        
        cols = DataVariant.SEMANTIC_DATA.column_names()
        # we construct via series to keep the datatype correct
        iter = zip(cols, object_data)
        return pd.DataFrame({ name : pd.Series(content) for name, content in iter})

    def split_by_class(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Splits the dataframe by class into a list of dataframes.
        The list of dataframes is sorted according to class_id.
        """
        group_by_class_id = {class_id: x for class_id, x in df.groupby(df['Class'])}
        return list(dict(sorted(group_by_class_id.items())).values())

    def split_rad_by_threshold(self, 
                               df: pd.DataFrame, 
                               static_object_doppler_threshold: float = 0.2) -> List[pd.DataFrame]:
        """
        Splits the dataframe by a threshold value for static object into two parts by adding a third dimension to the array. 
        The static objects are at index 0. The dynamic objects are at index 1.


        :param df: the RAD dataframe to be split
        :param static_object_doppler_threshold: the threshold value to split the dataframe into two

        Returns a list of dataframes, where the first contains static objects only and the second dynamic objects
        """
        mask = df['Doppler Compensated [m/s]'].abs() < static_object_doppler_threshold

        return [df[mask], df[~mask]]
    
    def _get_data_for_objects_in_frame(self, 
                                       loader: FrameDataLoader, 
                                       transforms: FrameTransformMatrix,
                                       labels: FrameLabels) -> Optional[List[np.ndarray]]:
        """
        For each object in the frame retrieve the following data: frame number, ranges, azimuths, 
        object class, summarized class, relative velocity compensated, number of detections, 
        bounding box volume, relative velocity (doppler), height, width, length, x, y, z.

        :param loader: the loader of the current frame
        :param transforms: the transformation matrix of the current frame
        :param labels: the labels of the current frame

        Returns: a list of arrays in the following order: frame number, ranges, azimuths, 
        object class, summarized class, relative velocity compensated, number of detections, 
        bounding box volume, relative velocity (doppler), height, width, length, x, y, z
        """
        
        radar_data_r = prepare_radar_data(loader)
        if radar_data_r is None:
            return None
        
        frame_numbers: List[np.ndarray] = []
        #object_ids: List[np.ndarray] = [] # TODO future work
        object_clazz: List[np.ndarray] = [] # this class stems from the dataset
        summarized_clazz: List[np.ndarray] = [] # we summarize multiple classes here for easier plotting
        dopplers_compensated: List[np.ndarray] = [] # avg doppler, but compensated for ego-vehicle movement, per object
        dopplers: List[np.ndarray] = [] # one avg doppler value per bounding box
        detections: List[np.ndarray] = [] # number of radar_points inside a bounding box
        bbox_vols: List[np.ndarray] = [] # bounding box volume in m^3
        ranges: List[np.ndarray] = [] # range in m
        azimuths: List[np.ndarray] = [] # azimuth in degrees
        elevations: List[np.ndarray] = [] # elevation in degrees
        
        # IMPORTANT: see docs/figures/Prius_sensor_setup_5.png (radar) for the directions of these variables
        x: List[np.ndarray] = []
        y: List[np.ndarray] = []
        z: List[np.ndarray] = []
        
        height: List[np.ndarray] = []
        width: List[np.ndarray] = []
        length: List[np.ndarray] = []
        
        radar_data_r = prepare_radar_data(loader)
        
        # each array is of shape (N, 7) conataining the points matching one label
        matching_points: List[Tuple[dict, Optional[np.ndarray]]] = find_matching_points_for_bboxes(radar_points=radar_data_r, labels=labels, transforms=transforms)
        
        for label, points_matching in matching_points:
            if points_matching is None:
                continue
            
            clazz_id = ex.get_class_id_from_name(label['label_class'], summarized=False)
            summarized_id = ex.convert_to_summarized_class_id(clazz_id)
            
            frame_numbers.append(loader.frame_number)
            object_clazz.append(clazz_id)
            summarized_clazz.append(summarized_id)
            dopplers_compensated.append(np.mean(points_matching[:, 5]))
            detections.append(points_matching.shape[0])
            bbox_vols.append(label['l'] * label['h'] * label['w'])         
            
            loc_camera = np.array([[label['x'], label['y'], label['z']]])
            
            # transform from camera coordinates to radar coordinates, stay cartesian
            # we take the center point of the object to calculate, its elevation and azimuth
            loc_radar = homogenous_transformation_cart(points=loc_camera, transform=transforms.t_radar_camera)
            range_from_loc = ex.locs_to_distance(loc_radar)
            ranges.append(range_from_loc)
            
            # look at Prius_sensor_setup_5 radar coordinate system to understand indexes in the next lines
            azimuths.append(ex.azimuth_angle_from_location(loc_radar[:, :2]))
            dopplers.append(np.mean(points_matching[:, 4]))
            elevations.append(ex.elevation_angle_from_location(loc_radar[:, [0, 2]]))
            
            x.append(loc_radar[0, 0])
            y.append(loc_radar[0, 1])
            z.append(loc_radar[0, 2])
            
            height.append(label['h'])
            width.append(label['w'])
            length.append(label['l'])
            
        
        if not object_clazz:
            return None
        
        columns = [frame_numbers, ranges, azimuths, dopplers, elevations, 
                   object_clazz, summarized_clazz, detections, bbox_vols, dopplers_compensated, 
                   height, width, length, x, y, z]
        return list(map(np.hstack, columns))
            
