import numpy as np

from extraction import points_in_bbox

class TestParameterExtraction():

    
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

        bbox = np.vstack([x_corners, y_corners, z_corners])
        
        radar_points = np.array([[1, 1, 1], [9, 8, 7], [-1, 1, 1], [1, 11, 0], [0, 0, 0], [10, 10, 10], [1, 1, -1]])
        
        inside_points_res = points_in_bbox(radar_points=radar_points, bbox=bbox)
        inside_points_expected = np.array([[1, 1, 1], [9, 8, 7], [0, 0, 0], [10, 10, 10]])
        
        assert np.array_equal(inside_points_res, inside_points_expected)
