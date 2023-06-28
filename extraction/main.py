
import sys
import os
import logging
from extraction.analysis_helper import prepare_data_analysis

from extraction.stats_table import generate_stats

sys.path.append(os.path.abspath("../view-of-delft-dataset"))

from extraction.file_manager import DataManager
from extraction.visualization import ParameterRangePlotter, PlotType
from vod.configuration.file_locations import KittiLocations


def set_logger():
    # https://stackoverflow.com/questions/14058453/making-python-loggers-output-all-messages-to-stdout-in-addition-to-log-file
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)

def main():
    set_logger()
    
    output_dir = "output"
    root_dir = "../view_of_delft_PUBLIC/"
    kitti_locations = KittiLocations(root_dir=root_dir,
                                     output_dir=output_dir,
                                     frame_set_path="",
                                     pred_dir="",
                                     )

    def abs(p): return os.path.abspath(p)
    logging.info(f"Radar directory: {abs(kitti_locations.radar_dir)}")
    logging.info(f"Label directory: {abs(kitti_locations.label_dir)}")
    logging.info(f"Output directory: {abs(kitti_locations.output_dir)}")

    dm = DataManager(kitti_locations=kitti_locations)
    plotter = ParameterRangePlotter(data_manager=dm)
    
    #generate_all_annotated_images(kitti_locations=kitti_locations)
    # after running this use:
    # ffmpeg -framerate 30 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p out.mp4
    # to convert to video (see https://stackoverflow.com/questions/24961127/how-to-create-a-video-from-images-with-ffmpeg)

    # plotter.plot_data_simple([PlotType.HISTOGRAM])
    #generate_stats(dm)
    
    #plotter.plot_syn_sem_combined()
    # plotter.plot_by_class_combined()
    plotter.plot_by_class_combined_main_only()
    # plotter.plot_rad()
    #plotter.plot_rade()
    # plotter.plot_heatmap()
    
    #prepare_data_analysis(dm)

if __name__ == '__main__':
    main()