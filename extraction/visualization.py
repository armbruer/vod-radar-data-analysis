# TODO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from enum import Enum
from typing import List
from extraction.extract import DataVariant, ParameterRangeExtractor
from vod.configuration.file_locations import KittiLocations
from vod.evaluation.evaluation_common import get_label_annotations
import extraction as ex
import sys
sys.path.append("/home/eric/Documents/mt/radar_dataset/view-of-delft-dataset")


class PlotType(Enum):
    VIOLIN = 1,
    BOXPLOT = 2,
    HISTOGRAM = 3,
    KNEEPLOT = 4


class ParameterRangePlotter:

    def now(_): return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    def __init__(self, kitti_locations: KittiLocations) -> None:
        self.kitti_locations = kitti_locations

    def plot_parameters(self, parameters: List[np.ndarray], titles: List[str], plot_types: List[PlotType], figure_name: str ='parameters') -> None:

        if len(parameters) != len(titles):
            raise ValueError(
                'Expecting equal amounts of parameters and titles')

        if plot_types[0] != PlotType.KNEEPLOT:
            for p in parameters:
                if p.ndim != 1:
                    raise ValueError(
                        'Expecting each parameter distribution to be of dimension 1')

        _, axs = plt.subplots(len(plot_types), len(parameters))

        for i, pt in enumerate(plot_types):
            for j, title in enumerate(titles):
                param = parameters[i*len(title)+j]
                axis = axs[i, j]

                axis.set_title(title)
                # TODO need x, y labels

                if pt == PlotType.VIOLIN:
                    sns.violinplot(x=param, ax=axis)
                elif pt == PlotType.BOXPLOT:
                    sns.boxplot(x=param, ax=axis)
                elif pt == PlotType.HISTOGRAM:
                    sns.histplot(x=param, ax=axis, bins=50)
                elif pt == PlotType.KNEEPLOT:
                    sns.lineplot(data=param)

        plt.tight_layout()
        plt.show()
        path = f"{self.kitti_locations.output_dir}/{figure_name}_{ParameterRangePlotter.now()}"
        plt.savefig(f'{path}.svg', format='svg')
        plt.savefig(f'{path}.png', format='png')
        plt.close()

    def plot_kneeplot(self, param: np.ndarray, title: str) -> None:
        if param.ndim != 1:
            raise ValueError(
                'Expecting each parameter distribution to be of dimension 1')

        param_sorted = np.sort(param)
        indices = np.arange(0, param.shape[0], 1)
        combined = np.hstack((indices, param_sorted))

        self.plot_parameters([combined], [title], [PlotType.KNEEPLOT])

    def plot_rad_data(self, rad, data_variant: DataVariant):
        titles = ["range (m)", "angle (degree)", "doppler (m/s)"]
        plot_types = [PlotType.BOXPLOT, PlotType.VIOLIN, PlotType.HISTOGRAM]
        extractor = ParameterRangeExtractor(self.kitti_locations)
        
        rad = list(extractor.get_data(data_variant).T)
        self.plot_parameters(params=rad, titles=titles, plot_types=plot_types)

    def plot_kneeplot_from_syntactic_data(self):
        extractor = ParameterRangeExtractor(self.kitti_locations)
        rad = extractor.get_data(DataVariant.SYNTACTIC_RAD)
        
        self.plot_kneeplot(param=rad[2], title='kneeplot doppler')


def main():
    output_dir = "output"
    root_dir = "/home/eric/Documents/mt/radar_dataset/view_of_delft_PUBLIC/"
    kitti_locations = KittiLocations(root_dir=root_dir,
                                     output_dir=output_dir,
                                     frame_set_path="",
                                     pred_dir="",
                                     )

    print(f"Lidar directory: {kitti_locations.lidar_dir}")
    print(f"Radar directory: {kitti_locations.radar_dir}")

    plotter = ParameterRangePlotter(kitti_locations=kitti_locations)
    plotter.plot_kneeplot_from_syntactic_data()
    plotter.plot_rad_data(DataVariant.SYNTACTIC_RAD)


if __name__ == '__main__':
    main()
