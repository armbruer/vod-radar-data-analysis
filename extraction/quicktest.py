# TODO
import sys
import os
sys.path.append(os.path.abspath("../view-of-delft-dataset"))

from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader


kitti_locations = KittiLocations(root_dir="example_set",
                                output_dir="example_output")

frame_data = FrameDataLoader(kitti_locations=kitti_locations,
                             frame_number="01201")



from vod.visualization import Visualization3D

vis3d = Visualization3D(frame_data)

vis3d.draw_plot(    radar_origin_plot = True,
                  lidar_origin_plot = True,
                  camera_origin_plot = True,
                  lidar_points_plot = True,
                  radar_points_plot = True,
                  radar_velocity_plot = True,
                  annotations_plot = True)