import os

from tqdm import tqdm
from typing import List
from extraction.helpers import get_class_list
from vod.common.file_handling import get_frame_list_from_folder
from vod.configuration.file_locations import KittiLocations
from vod.frame.data_loader import FrameDataLoader
from vod.visualization.vis_2d import Visualization2D

def generate_annotated_images(kitti_locations: KittiLocations, 
                              outdir: str, 
                              lidar: bool = True, 
                              radar: bool = True, 
                              classes_visualized: List[str]=get_class_list()):
    
    dir = f'{kitti_locations.output_dir}/{outdir}'
    os.makedirs(dir, exist_ok=True)

    frames = get_frame_list_from_folder(kitti_locations.label_dir, '.txt')
    for frame_number in tqdm(frames, desc="Generating annotated images"):
        if os.path.exists(f'{dir}/{frame_number}.png'):
            continue
        
        loader = FrameDataLoader(
            kitti_locations=kitti_locations, frame_number=frame_number)

        vis2d = Visualization2D(frame_data_loader=loader, classes_visualized=classes_visualized)
        vis2d.draw_plot(plot_figure=False, save_figure=True, show_gt=True,
                        show_lidar=lidar, show_radar=radar, outdir=outdir)
        
        
def generate_all_annotated_images(kitti_locations: KittiLocations):
    generate_annotated_images(kitti_locations=kitti_locations, outdir=f"images_annoted_radar_only/", lidar=False)
    generate_annotated_images(kitti_locations=kitti_locations, outdir=f"images_annotated/")
