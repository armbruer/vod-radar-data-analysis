from typing import Tuple
import numpy as np
import pytest
from extraction.file_manager import DataManager, DataView
from extraction.helpers import DataVariant, DataViewType, find_matching_points_for_bboxes, get_bbox_transformation_matrix, locs_to_distance, points_in_bbox, prepare_radar_data

from vod.configuration.file_locations import KittiLocations
from vod.frame.data_loader import FrameDataLoader
from vod.frame.labels import FrameLabels
from vod.frame.transformations import FrameTransformMatrix, homogenous_transformation_cart
from vod.visualization.helpers import get_3d_label_corners

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
def data_manager(kitti_locations: KittiLocations):
    data_manager = DataManager(kitti_locations=kitti_locations)
    yield data_manager
    
@pytest.fixture()
def object_sample(kitti_locations: KittiLocations):
    # extracted labels dict for a single object
    labels_dict_object = [{'label_class': 'rider', 'h': 1.5620380218093075, 'w': 0.8727467348937346, 'l': 0.7789294078511312, 
                         'x': 1.7167627537059729, 'y': 1.3749588244970534, 'z': 2.060029085851716, 'rotation': -1.880353040241324, 'score': 1.0}]
        
    labels = FrameLabels([])
    labels._labels_dict = labels_dict_object
    
    loader = FrameDataLoader(kitti_locations=kitti_locations, frame_number='01211')
    transforms = FrameTransformMatrix(loader)
    
    yield labels, loader, transforms

# https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
def cart2pol(x, y):
    r = np.sqrt(x**2 + y**2)
    angle = np.arctan2(y, x) * 180 / np.pi
    return r, angle

def pol2cart(r, angle):
    angle = angle * np.pi / 180
    x = r * np.cos(angle)
    y = r * np.sin(angle)
    return x, y

def equals(a1: np.ndarray, a2: np.ndarray) -> bool:
    return np.allclose(a1, a2)
    

class TestHelpers():
    
    def test_corners_layout(self, object_sample: Tuple[FrameLabels, FrameDataLoader, FrameTransformMatrix]):
        # quick sanity check that my understanding of corners layout order is correct
        
        # order of corners
        #    5--------4 
        #   /|       /| | height (z)
        #  / |      / | |
        # 6--------7  | |
        # |  |     |  |
        # |  1-----|--0 ^ length (x)
        # | /      | / /
        # |/       |/ /
        # 2--------3 ---> width (y)
        
        labels, _, _ = object_sample
        labels_dict = labels.labels_dict
        
        corners = get_3d_label_corners(labels)[0]['corners_3d'].T
        
        assert abs(corners[4][2] - corners[0][2]) == labels_dict[0]['h']
        assert abs(corners[1][1] - corners[0][1]) == labels_dict[0]['w'] 
        assert abs(corners[1][0] - corners[3][0]) == labels_dict[0]['l']
        
        assert abs(corners[6][1] - corners[7][1]) == labels_dict[0]['w'] 
        assert abs(corners[6][0] - corners[5][0]) == labels_dict[0]['l']
        assert abs(corners[6][2] - corners[3][2]) == labels_dict[0]['h']
        
        assert corners[5][0] == corners[4][0] == corners[1][0] == corners[0][0]
        assert corners[3][0] == corners[2][0] == corners[6][0] == corners[7][0]
        
        assert corners[3][1] == corners[0][1] == corners[7][1] == corners[4][1]
        assert corners[1][1] == corners[2][1] == corners[5][1] == corners[6][1]
        
        assert corners[5][2] == corners[4][2] == corners[6][2] == corners[7][2]
        assert corners[1][2] == corners[2][2] == corners[3][2] == corners[0][2]
        
    def test_points_in_bbox(self):

        # order of corners
        #    5--------4 
        #   /|       /| | height (z)
        #  / |      / | |
        # 6--------7  | |
        # |  |     |  |
        # |  1-----|--0 ^ length (x)
        # | /      | / /
        # |/       |/ /
        # 2--------3 ---> width (y)

        x_corners = [10, 10, 0, 0, 10, 10, 0, 0]
        y_corners = [10, 0, 0, 10, 10, 0, 0, 10]
        z_corners = [0, 0, 0, 0, 10, 10, 10, 10]

        bbox = np.vstack([x_corners, y_corners, z_corners]).T
        
        radar_points = np.array([[1, 1, 1], [9, 8, 7], [-1, 1, 1], [1, 11, 0], [0, 0, 0], [10, 10, 10], [1, 1, -1]])
        bbox_limits = np.array([10, 10 , 10])
        
        transformation = get_bbox_transformation_matrix(bbox)
        inside_points_res = points_in_bbox(radar_points=radar_points, bbox_limits=bbox_limits, transformation_matrix=transformation)
        inside_points_expected = np.array([[1, 1, 1], [9, 8, 7], [0, 0, 0], [10, 10, 10]])
        
        assert np.array_equal(inside_points_res, inside_points_expected)
        
    def test_transformations(self, object_sample: Tuple[FrameLabels, FrameDataLoader, FrameTransformMatrix]):
        labels, _, transforms = object_sample
        labels_dict = labels.labels_dict[0]
        
        loc_c = np.array([[labels_dict['x'], labels_dict['y'], labels_dict['z']]])
        
        loc_r = homogenous_transformation_cart(points=loc_c, transform=transforms.t_radar_camera)
        loc_c2 = homogenous_transformation_cart(points=loc_r, transform=transforms.t_camera_radar)
        
        assert np.array_equal(loc_c, loc_c2)
    
    def test_azimuth_elevation_calculation(self, data_manager: DataManager):
        for dv in DataVariant.all_variants():
            data_view: DataView = data_manager.get_view(dv, DataViewType.NONE)
            dfs = data_view.df
            if not isinstance(dfs, list):
                dfs = [dfs]
                
            for df in dfs:
                df = df.head(n=5)
                
                range_exp = df['Range [m]'].to_numpy().T
                az_exp = df['Azimuth [degree]'].to_numpy().T
                ev_exp = df['Elevation [degree]'].to_numpy().T
                
                # split into three separate numpy arrays
                loc_radar = df[['x', 'y', 'z']].to_numpy()
                x_radar, y_radar, z_radar = map(np.array, list(loc_radar.T))
                
                assert equals(range_exp,  locs_to_distance(loc_radar))
                r, az = cart2pol(x_radar, -y_radar)
                assert equals(az, az_exp)
                
                x, y = pol2cart(r, az)
                assert equals(x, x_radar)
                assert equals(y, -y_radar)
  
                r, ev = cart2pol(x_radar, z_radar)
                assert equals(ev, ev_exp)
                
                x, z = pol2cart(r, ev)
                assert equals(x, x_radar)
                assert equals(z, z_radar)
                
                
    def test_points_in_bbox_with_different_coordinate_systems(self, 
                                                              kitti_locations: KittiLocations):
        loader = FrameDataLoader(kitti_locations=kitti_locations, frame_number='04716')
        transforms = FrameTransformMatrix(loader)
        
        labels = FrameLabels(loader.get_labels())
        
        radar_data_r = prepare_radar_data(loader)
        if radar_data_r is None:
            assert False
            
        radar_data_c = homogenous_transformation_cart(radar_data_r[:, :3], transform=transforms.t_camera_radar)
        radar_data_c = np.hstack((radar_data_c, radar_data_r[:, 3:]))
        
        
        matching_points_r = find_matching_points_for_bboxes(radar_points=radar_data_r, 
                                                            labels=labels,
                                                            transforms=transforms)
        
        matching_points_c = find_matching_points_for_bboxes(radar_points=radar_data_c, 
                                                            labels=labels, 
                                                            transforms=transforms, camera_coordinates=True)
        
        assert len(matching_points_c) == len(matching_points_r)
        
        points_only = lambda matching_points: [e[1][:, :3] for e in matching_points if e[1] is not None]
        
        
        matching_points_exp = np.vstack(points_only(matching_points_r))
        matching_points_c = np.vstack(points_only(matching_points_c))
        matching_points_res = homogenous_transformation_cart(matching_points_c, transform=transforms.t_camera_radar)
        
        assert np.allclose(matching_points_exp, matching_points_res)
        
        
        
        
                
    def test_stuff(self):
        # code below generated with bing ai

        # Test case 1: No points inside the bounding box
        radar_points = np.array([[0, 0, -1], [1, 1, 1], [2, 2, 2]])
        bbox = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 0, 1], [0, 1, 1], [0, 1, 0], [0, 0, 0], [0, 0, 1]])
        bbox_limits = np.array([1, 1, 1])
        transformation_matrix = get_bbox_transformation_matrix(bbox)
        assert points_in_bbox(radar_points, bbox_limits, transformation_matrix) is None

        # # Test case 1: Some points inside the bbox
        # radar_points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        # bbox = np.array([[1, 1, 1], [0.5, 1.5, 0.5], [0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [1.5, 1.5, 1.5], [0.5, 1.5, 1.5], [0.5, 0.5, 1.5], [1.5, 0.5, 1.5]])
        # bbox_limits = np.array([np.sqrt(2), np.sqrt(2), np.sqrt(2)])
        # transformation_matrix = get_bbox_transformation_matrix(bbox)
        # assert points_in_bbox(radar_points,bbox_limits , transformation_matrix) is None

        # import numpy as np

        # Test case: Some points inside the bounding box, some points outside
        # radar_points = np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5], [0.7, 0.7, 0.7], [1.3, 1.3, 1.3]])
        # bbox = np.array([[1, 1, 1], [0.5, 1.5, 0.5], [0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [1.5, 1.5, 1.5], [0.5, 1.5, 1.5], [0.5, 0.5, 1.5], [1.5, 0.5, 1.5]])
        # bbox_limits = np.array([np.sqrt(2), np.sqrt(2), np.sqrt(2)])
        # transformation_matrix = get_bbox_transformation_matrix(bbox)
        # expected_result = np.array([[0.7 ,0.7 ,0.7],[1.3 ,1.3, 1.3]])
        # assert points_in_bbox(radar_points,bbox_limits , transformation_matrix) == expected_result

        
        radar_points = np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5], [0.7, 0.7, 0.7], [1.3, 1.3, 1.3]])
        bbox = np.array([[1, 1, 2], [2, 2, 3], [3, 1, 2], [2, 0, 1], [1, 1, 4], [2, 2, 5], [3, 1, 4], [2, 0, 3]])
        bbox_limits = np.array([np.sqrt(2), np.sqrt(2), np.sqrt(2)])
        transformation_matrix = get_bbox_transformation_matrix(bbox)
        test = homogenous_transformation_cart(bbox, transformation_matrix)
        expected_result = np.array([[0.7 ,0.7 ,0.7],[1.3 ,1.3, 1.3]])
        assert points_in_bbox(radar_points,bbox_limits , transformation_matrix) == expected_result
                