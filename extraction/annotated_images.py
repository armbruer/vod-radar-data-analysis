import os
import multiprocessing

from itertools import repeat
from typing import List
from extraction.helpers import get_class_names
from vod.common.file_handling import get_frame_list_from_folder
from vod.configuration.file_locations import KittiLocations
from vod.frame.data_loader import FrameDataLoader
from vod.visualization.vis_2d import Visualization2D


def gen_annotation(frame_number: str, dir: str, kitti_locations: KittiLocations, modalities: List[bool], subdir):
    if os.path.exists(f'{dir}/{frame_number}.png'):
        return
        
    lidar, radar = modalities
    loader = FrameDataLoader(
        kitti_locations=kitti_locations, frame_number=frame_number)

    vis2d = Visualization2D(frame_data_loader=loader, classes_visualized=get_class_names(False))
    vis2d.draw_plot(plot_figure=False, save_figure=True, show_gt=True,
                    show_lidar=lidar, show_radar=radar, subdir=subdir)

def generate_annotated_images(kitti_locations: KittiLocations, 
                              subdir: str, 
                              lidar: bool = True, 
                              radar: bool = True):
    
    dir = f'{kitti_locations.output_dir}/{subdir}'
    os.makedirs(dir, exist_ok=True)

    frames = get_frame_list_from_folder(kitti_locations.label_dir, '.txt')
    iter = zip(frames, repeat(dir), repeat(kitti_locations), repeat([lidar, radar]), repeat(subdir))
    
    cpus = int(multiprocessing.cpu_count() * 0.75)
    try:
        pool = multiprocessing.Pool(processes=cpus)
        pool.starmap(gen_annotation, iter)
    finally:
        pool.close()
        pool.join()
        
        
def generate_all_annotated_images(kitti_locations: KittiLocations):
    generate_annotated_images(kitti_locations=kitti_locations, subdir=f"images_annoted_radar_only", lidar=False)
    generate_annotated_images(kitti_locations=kitti_locations, subdir=f"images_annotated")

