
import sys
import os
import logging
from extraction import visualization
from extraction.annotated_images import generate_all_annotated_images
from extraction.helpers import DataVariant

# python made some shenanigans without this :/
sys.path.append(os.path.abspath("../delt-radar-data-extractor"))

from extraction.analysis_helper import azi_large_filter, investigate_azimuth, prepare_data_analysis
from extraction.stats_table import generate_stats
from extraction.file_manager import DataManager
from extraction.plotting import DistributionPlotter
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
    
def overall_dataset_stats(dm: DataManager):
    sem_view = dm.get_view(DataVariant.SEMANTIC_DATA)
    syn_view = dm.get_view(DataVariant.SYNTACTIC_DATA)
    
    logging.info("Overall dataset stats: ")
    logging.info(f"Semantic Data Length: {sem_view.df.shape[0]}")
    logging.info(f"Syntactic Data Length: {syn_view.df.shape[0]}")
    
def investigate_azimuth_anomaly(kitti_locations: KittiLocations, dm: DataManager):
    logging.info("Generating data for investigating azimuth anomalies...")
    
    # this is used to investiage particular azimuth anomalies
    visualization.visualize_frame_sequence(
         data_variant=DataVariant.SEMANTIC_DATA,
         kitti_locations=kitti_locations,
         min_frame_number=7000, 
         max_frame_number=7020,
         tracking_id=161)
    investigate_azimuth(dm)
    investigate_azimuth(dm, 'azimuth_large', filter=azi_large_filter)
    
def annotate_all_images(kitti_locations: KittiLocations):
    # apply these steps to generate the video sequences with annotations overlaid
    # 
    generate_all_annotated_images(kitti_locations=kitti_locations)
    # 
    # after running this use:
    # ffmpeg -framerate 30 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p out.mp4
    # to convert to video (see https://stackoverflow.com/questions/24961127/how-to-create-a-video-from-images-with-ffmpeg)

    
def run_mt_analysis(plotter: DistributionPlotter, 
                    dm: DataManager,
                    stats = True,
                    syn_sem_combined = True,
                    data_analysis_helper = True,
                    by_class = True,
                    rade = True,
                    azimuth_heatmap = True,
                    elevation_heatmap = True):
    # these analyses are actually used in the master thesis
    # I only verified these work in the final commit
    
    if stats:
        logging.info("Generating basic statistical measures...")
        generate_stats(dm)
    
    if data_analysis_helper:
        prepare_data_analysis(dm, variants=DataVariant.basic_variants())
    
    if syn_sem_combined:
        plotter.plot_syn_sem_combined()
    
    if by_class:
        logging.info("Generating per class RADE distribution plots...")
        plotter.plot_by_class_combined(most_important_only=True)
    
    if rade:
        logging.info("Generating overall RADE distribution plots...")
        plotter.plot_rade()
    
    if azimuth_heatmap:
        logging.info("Generating azimuth heatmaps for selected classes...")
        plotter.plot_azimuth_heatmap()
    
    if elevation_heatmap:
        logging.info("Generating elevation heatmaps for selected classes...")
        plotter.plot_ele_heatmap()
    
def run_additional_analysis(plotter: DistributionPlotter):
    # these stats below are not used in the master thesis, don't expect them to run...
    # 
    # 
    plotter.correlation_heatmap(data_variant=DataVariant.SEMANTIC_DATA)
    plotter.plot_data_simple_improved(data_variants=DataVariant.all_variants())
    
    
    # this is our try at applying grid search to find the best hyperparameters for KDE
    # something is off, cause it takes forever...
    # 
    plotter.plot_all_kdeplots()

def main():
    set_logger()
    
    # all output is generated in this directory
    output_dir = "output"
    # the dataset is expected here
    root_dir = "../view_of_delft_PUBLIC/"
    kitti_locations = KittiLocations(root_dir=root_dir,
                                     output_dir=output_dir,
                                     frame_set_path="",
                                     pred_dir="",
                                     )

    def abs(p): return os.path.abspath(p)
    logging.info(f"Radar dataset directory: {abs(kitti_locations.radar_dir)}")
    logging.info(f"Label dataset directory: {abs(kitti_locations.label_dir)}")
    logging.info(f"Output directory: {abs(kitti_locations.output_dir)}")

    dm = DataManager(kitti_locations=kitti_locations)
    

    plotter = DistributionPlotter(data_manager=dm)
    overall_dataset_stats(dm)

    run_mt_analysis(plotter, dm, False, False, True, False, False, False, False)

if __name__ == '__main__':
    main()