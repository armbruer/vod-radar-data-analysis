from typing import List
import numpy as np
import pytest
import pandas as pd
from extraction.file_manager import DataManager, DataView
from extraction.helpers import DataVariant, DataViewType, locs_to_distance, points_in_bbox

from vod.configuration.file_locations import KittiLocations
from vod.frame.data_loader import FrameDataLoader
from vod.frame.transformations import FrameTransformMatrix, homogenous_transformation_cartesian_coordinates

@pytest.fixture()
def kitti_locations():
    output_dir = "output"
    root_dir = "../view_of_delft_PUBLIC/"
    kitti_locations = KittiLocations(root_dir=root_dir,
                                    output_dir=output_dir,
                                    frame_set_path="",
                                    pred_dir="",
                                    )
    yield kitti_locations

@pytest.fixture()
def data_manager(kitti_locations):
    data_manager = DataManager(kitti_locations=kitti_locations)
    yield data_manager


# https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
def cart2pol(x, y):
    r = np.sqrt(x**2 + y**2)
    angle = np.arctan2(y, x) * 180 / np.pi
    return pd.Series(r), pd.Series(angle)

def pol2cart(r, angle):
    angle = angle * np.pi / 180
    x = r * np.cos(angle)
    y = r * np.sin(angle)
    return pd.Series(x), pd.Series(y)

def radar_to_camera_loc(kitti_locations, df, loc_radar):
    
    loc_camera_list: List[np.ndarray] = []
    
    # they do not neccessarily all have the same frame number, so this is kind of complicated...
    for i, (_, series) in enumerate(df.iterrows()):
        fn = series['Frame Number']
        loader = FrameDataLoader(kitti_locations=kitti_locations, frame_number=fn)
        transforms = FrameTransformMatrix(frame_data_loader_object=loader)
        loc_camera = homogenous_transformation_cartesian_coordinates(np.atleast_2d(loc_radar[i]), transforms.t_camera_radar)
        
        loc_camera_list.append(loc_camera)
        
    return np.vstack(loc_camera_list)


def equals(s1: pd.Series, s2: pd.Series) -> bool:
    return s1.round(1).eq(s2.round(1)).all()

    

class TestHelpers():
    
    def test_points_in_bbox(self):

        # order of corners
        #    7--------4
        #   /|       /|
        #  / |      / |
        # 6--------5  |
        # |  |     |  |
        # |  3-----|--0
        # | /      | /
        # |/       |/
        # 2--------1
        # but the origin is at 2!

        x_corners = [10, 10, 0, 0, 10, 10, 0, 0]
        y_corners = [10, 0, 0, 10, 10, 0, 0, 10]
        z_corners = [0, 0, 0, 0, 10, 10, 10, 10]

        bbox = np.vstack([x_corners, y_corners, z_corners]).T
        
        radar_points = np.array([[1, 1, 1], [9, 8, 7], [-1, 1, 1], [1, 11, 0], [0, 0, 0], [10, 10, 10], [1, 1, -1]])
        
        inside_points_res = points_in_bbox(radar_points_radar=radar_points, radar_points_camera=radar_points bbox=bbox)
        inside_points_expected = np.array([[1, 1, 1], [9, 8, 7], [0, 0, 0], [10, 10, 10]])
        
        assert np.array_equal(inside_points_res, inside_points_expected)
    
    def test_azimuth_elevation_calculation(self, data_manager: DataManager, kitti_locations: KittiLocations):
        # TODO: Fix the other data variants!!!!
        for dv in [DataVariant.SEMANTIC_DATA]:
            data_view: DataView = data_manager.get_view(dv, DataViewType.NONE)
            dfs = data_view.df
            if not isinstance(dfs, list):
                dfs = [dfs]
                
            for df in dfs:
                df = df.head(n=5)
                
                range_exp = df['Range [m]']
                az_exp = df['Azimuth [degree]']
                ev_exp = df['Elevation [degree]']
                loc_radar = np.array([df['x'], df['y'], df['z']]).T
                
                camera_loc = radar_to_camera_loc(kitti_locations, df, loc_radar)
                x_camera, y_camera, z_camera = list(camera_loc.T)
                
                assert range_exp.eq( locs_to_distance(loc_radar)).all()
                r, az = cart2pol(z_camera, x_camera)
                assert equals(az, az_exp)
                
                z, x = pol2cart(r, az)
                assert equals(z, z_camera)
                assert equals(x, x_camera)
  
                r, ev = cart2pol(z_camera, y_camera)
                assert equals(ev, ev_exp)
                
                z, y = pol2cart(r, ev)
                assert equals(z, z_camera)
                assert equals(y, y_camera)
                    