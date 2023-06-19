import sys
import os

sys.path.append(os.path.abspath("../view-of-delft-dataset"))

from typing import List
from extraction.helpers import get_class_list
from vod.common.file_handling import get_frame_list_from_folder
from vod.configuration.file_locations import KittiLocations

from tqdm import tqdm
from vod.frame.data_loader import FrameDataLoader
from vod.visualization.vis_2d import Visualization2D


def abspath(path): return os.path.abspath(path)


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
        
        

def main():
    output_dir = "output"
    root_dir = "../view_of_delft_PUBLIC/"
    kitti_locations = KittiLocations(root_dir=root_dir,
                                     output_dir=output_dir,
                                     frame_set_path="",
                                     pred_dir="",
                                     )

    output_dir = abspath(kitti_locations.output_dir)
    print(f"Output directory: {output_dir}")
    
    #generate_annotated_images(kitti_locations=kitti_locations, outdir=f"images_annoted_radar_only/", lidar=False)
    generate_annotated_images(kitti_locations=kitti_locations, outdir=f"images_annotated/")

if __name__ == '__main__':
    main()
