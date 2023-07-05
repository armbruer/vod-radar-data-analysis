import numpy as np
import pandas as pd

from tqdm import tqdm
from extraction.helpers import DataVariant
import extraction as ex
from vod.configuration.file_locations import KittiLocations
from vod.frame import FrameTransformMatrix
from vod.frame import FrameDataLoader
from vod.common.file_handling import get_frame_list_from_folder
from typing import Dict, List, Optional

from vod.frame.labels import FrameLabels
from vod.frame.transformations import homogenous_transformation_cartesian_coordinates
from vod.visualization.helpers import get_placed_3d_label_corners

class ParameterRangeExtractor:

    def __init__(self, kitti_locations: KittiLocations) -> None:
        self.kitti_locations = kitti_locations

    def extract_data_from_syntactic_data(self) -> pd.DataFrame:
        """
        Extract the frame number, range, azimuth, doppler values and loactions for each detection in this dataset.
        This method works on the syntactic (unannotated) data of the dataset.


        Returns a dataframe with columns frame_number, range, azimuth, doppler, relative velocity (doppler) compensated, x, y, z
        """
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
        
        # TODO future work: rcs
        for frame_number in tqdm(iterable=frame_numbers, desc='Syntactic data: Going through frames'):
            loader = FrameDataLoader(
                kitti_locations=self.kitti_locations, frame_number=frame_number)

            # radar_data shape: [x, y, z, RCS, v_r, v_r_compensated, time] (-1, 7)
            radar_data = loader.radar_data
            if radar_data is not None:
                # flip the y-axis, so left of 0 is negative and right is positive as one would expect it in plots
                # see docs/figures/Prius_sensor_setup_5.png (radar)
                # this is required for azimuth and elevation calculation
                radar_data[:,1] = -radar_data[:, 1]
                
                frame_nums.append(np.full(radar_data.shape[0], frame_number))
                ranges.append(ex.locs_to_distance(radar_data[:, :3]))
                azimuths.append(ex.azimuth_angle_from_location(radar_data[:, :2]))
                elevations.append(ex.elevation_angle_from_location(radar_data[:, [1, 3]]))
                dopplers.append(radar_data[:, 4])
                dopplers_compensated.append(radar_data[:, 5])
                
                # unflip the y coordinate
                radar_data[:,1] = -radar_data[:, 1]
                
                x.append(radar_data[:, 0])
                y.append(radar_data[:, 1])
                z.append(radar_data[:, 2])

        columns = [frame_nums, ranges, azimuths, dopplers, elevations, dopplers_compensated, x, y, z]
        data = list(map(np.hstack, columns))
        col_names = DataVariant.SYNTACTIC_DATA.column_names()
        # we construct via series to keep the datatype correct
        return pd.DataFrame({ name : pd.Series(content) for name, content in zip(col_names, data)})

    def extract_object_data_from_semantic_data(self) -> pd.DataFrame:
        """
        For each object in the frame retrieve the following data: frame number, ranges, azimuths, 
        object class, summarized class, relative velocity compensated, number of detections, 
        bounding box volume, relative velocity (doppler), height, width, length, x, y, z.

        :param loader: the loader of the current frame
        :param transforms: the transformation matrix of the current frame

        Returns: a pandas dataframe with the following columns: frame number, ranges, azimuths, 
        object class, summarized class, relative velocity compensated, number of detections, 
        bounding box volume, relative velocity (doppler), height, width, length, x, y, z
        """

        # only those frame_numbers which have annotations
        frame_numbers = get_frame_list_from_folder(
            self.kitti_locations.label_dir)

        object_data_dict: Dict[int, List[np.ndarray]] = {}

        for frame_number in tqdm(iterable=frame_numbers, desc='Semantic data: Going through frames'):
            loader = FrameDataLoader(
                kitti_locations=self.kitti_locations, frame_number=frame_number)
            transforms = FrameTransformMatrix(frame_data_loader_object=loader)

            object_data = self._get_data_for_objects_in_frame(
                loader=loader, transforms=transforms)
            if object_data is not None:
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
                                       transforms: FrameTransformMatrix) -> Optional[List[np.ndarray]]:
        """
        For each object in the frame retrieve the following data: frame number, ranges, azimuths, 
        object class, summarized class, relative velocity compensated, number of detections, 
        bounding box volume, relative velocity (doppler), height, width, length, x, y, z.

        :param loader: the loader of the current frame
        :param transforms: the transformation matrix of the current frame

        Returns: a list of arrays in the following order: frame number, ranges, azimuths, 
        object class, summarized class, relative velocity compensated, number of detections, 
        bounding box volume, relative velocity (doppler), height, width, length, x, y, z
        """
        
        labels = loader.get_labels()
        if labels is None:
            return None
        
        labels = FrameLabels(labels)
        
        # Step 1: Obtain corners of bounding boxes and radar data points
        labels_with_corners = get_placed_3d_label_corners(labels)
        
        # radar_points shape: [x, y, z, RCS, v_r, v_r_compensated, time] (-1, 7)
        radar_data = loader.radar_data
        if radar_data is None:
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
        
        # Step 2: Transform points into the same coordinate system as the labels
        radar_points_tr = homogenous_transformation_cartesian_coordinates(points=radar_data[:, :3], 
                                                                          transform=transforms.t_camera_radar)
        radar_points_tr = np.hstack((radar_points_tr, radar_data[:, 3:]))
        
        for label in labels_with_corners:
            # Step 3: For each bounding box get a list of radar points which are inside of it
            bbox = label['corners_3d_placed']
            points_matching = ex.points_in_bbox(radar_points_radar=radar_data, 
                                                radar_points_camera=radar_points_tr, 
                                                bbox=bbox)
            
            if points_matching is not None:
                points_matching = np.vstack(points_matching)
                clazz_id = ex.get_class_id_from_name(label['label_class'], summarized=False)
                summarized_id = ex.convert_to_summarized_class_id(clazz_id)
                
                # Step 4: Get the avg doppler value of the object and collect it
                frame_numbers.append(loader.frame_number)
                object_clazz.append(clazz_id)
                summarized_clazz.append(summarized_id)
                dopplers_compensated.append(np.mean(points_matching[:, 5]))
                detections.append(points_matching.shape[0])
                bbox_vols.append(label['l'] * label['h'] * label['w'])         
                
                loc = np.array([[label['x'], label['y'], label['z']]])
                
                # transform from camera coordinates to radar coordinates, stay cartesian
                loc_radar = homogenous_transformation_cartesian_coordinates(points=loc, 
                                                                            transforms=transforms.t_radar_camera)
                range_from_loc = ex.locs_to_distance(loc_radar)            
                ranges.append(range_from_loc)
                
                # DO NOT USE radar_coordinates to calculate the azimuth and elevation
                # The problem: camera located behind radar
                # look at Prius_sensor_setup_5 camera coordinate system to understand indexes in the next lines
                # x is already mirrored, no reason to mirror it
                azimuths.append(ex.azimuth_angle_from_location(loc[:, [2, 0]]))
                dopplers.append(np.mean(points_matching[:, 4]))
                elevations.append(ex.elevation_angle_from_location(loc[:, [2, 1]]))
                
                # we need to use radar coordinates here to stay in one coordinate system between different data variants
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