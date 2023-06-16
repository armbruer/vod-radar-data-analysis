import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # do not show figures when saving plot
import sys
import os

sys.path.append(os.path.abspath("../view-of-delft-dataset"))

from tqdm import tqdm
from vod.configuration.file_locations import KittiLocations
from extraction.stats_table import StatsTableGenerator
from extraction import DataVariant, ParameterRangeExtractor
from typing import List
from enum import Enum
from datetime import datetime

class PlotType(Enum):
    VIOLIN = 1,
    BOXPLOT = 2,
    HISTOGRAM = 3,
    KNEEPLOT = 4


class ParameterRangePlotter:

    def __init__(self, kitti_locations: KittiLocations) -> None:
        self.kitti_locations = kitti_locations

    def plot_data_simple(self, data_variant: DataVariant) -> None:
        plot_types = [PlotType.BOXPLOT, PlotType.VIOLIN, PlotType.HISTOGRAM]
        extractor = ParameterRangeExtractor(self.kitti_locations)
        data = extractor.get_data(data_variant)
        columns = data_variant.column_names(with_unit=True)

        self.plot_data(data=data, plot_types=plot_types,
                       data_variant=data_variant, value_labels=columns)

    def plot_data(self,
                  data: List[np.ndarray],
                  plot_types: List[PlotType],
                  data_variant: DataVariant,
                  **kwargs) -> None:

        cols = data[0].shape[1]
        figure_name = kwargs.get('figure_name', data_variant.name.lower())
        value_labels = kwargs.get('value_labels', cols * [''])
        other_labels = kwargs.get('other_labels', cols * [''])

        if (not (len(value_labels) == len(other_labels) == cols)):
            raise ValueError(
                f'Expecting the length of value_labels, other_labels to be equal to {cols}')

        for k, d in enumerate(data):
            index_name = data_variant.index_to_str(k)

            figure, axs = plt.subplots(len(value_labels), len(
                plot_types), figsize=(6.4, 10), layout='constrained')

            for i, value_label in tqdm(enumerate(value_labels), desc="Preparing subplots"):
                for j, pt in tqdm(enumerate(plot_types), desc="Going through plot types"):
                    param: np.ndarray = d[:, i]
                    other_label = other_labels[i]

                    if len(plot_types) > 1 and len(value_labels) > 1:
                        axis = axs[i, j]
                    elif len(plot_types) == 1 and len(value_labels) == 1:
                        axis = axs
                    elif len(plot_types) == 1:
                        axis = axs[j]  # the other index
                    else:
                        axis = axs[i]  # the other index

                    if pt == PlotType.VIOLIN:
                        gfg = sns.violinplot(y=param, ax=axis)
                        gfg.set(ylabel=value_label)
                    elif pt == PlotType.BOXPLOT:
                        gfg = sns.boxplot(y=param, ax=axis)
                        gfg.set(ylabel=value_label)
                    elif pt == PlotType.HISTOGRAM:
                        gfg = sns.histplot(x=param, ax=axis, bins=30)
                        axis.set_yscale('log')
                        gfg.set(xlabel=value_label)
                    elif pt == PlotType.KNEEPLOT:
                        indices = np.arange(0, param.shape[0], 1)
                        gfg = sns.lineplot(x=indices, y=param)
                        axis.set_xscale('log')
                        axis.grid()
                        gfg.set(xlabel=other_label, ylabel=value_label)

            # plt.show()

            figures_dir = f"{self.kitti_locations.figures_dir}/{data_variant.name.lower()}"
            os.makedirs(figures_dir, exist_ok=True)

            now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            path = f"{figures_dir}/{figure_name}_{index_name}_{now}"
            # figure.savefig(f'{path}.svg', format='svg')
            figure.savefig(f'{path}.png', format='png')
            logging.info(f'Plot generated in file:///{path}.png')

    def plot_kneeplot(self, param: np.ndarray, **kwargs) -> None:
        if param.ndim != 1:
            raise ValueError(
                'Expecting each parameter distribution to be of dimension 1')

        self.plot_data([np.sort(param)], [PlotType.KNEEPLOT], **kwargs)

    def plot_kneeplot_from_syntactic_data(self) -> None:
        extractor = ParameterRangeExtractor(self.kitti_locations)
        rad = extractor.get_data(DataVariant.SYNTACTIC_RAD)

        kwargs = {
            'value_labels': ['doppler (m/s)'],
            'other_labels': ['index'],
            'figure_name':  'kneeplot',
        }
        self.plot_kneeplot(param=rad[:, 2], **kwargs)


def main():
    output_dir = "output"
    root_dir = "../view_of_delft_PUBLIC/"
    kitti_locations = KittiLocations(root_dir=root_dir,
                                     output_dir=output_dir,
                                     frame_set_path="",
                                     pred_dir="",
                                     )

    def abs(p): return os.path.abspath(p)
    print(f"Radar directory: {abs(kitti_locations.radar_dir)}")
    print(f"Label directory: {abs(kitti_locations.label_dir)}")
    print(f"Output directory: {abs(kitti_locations.output_dir)}")

    plotter = ParameterRangePlotter(kitti_locations=kitti_locations)
    stats_generator = StatsTableGenerator(kitti_locations=kitti_locations)

    dvs = [DataVariant.SEMANTIC_RAD, DataVariant.SYNTACTIC_RAD, DataVariant.STATIC_DYNAMIC_RAD,
           DataVariant.SEMANTIC_OBJECT_DATA, DataVariant.SEMANTIC_OBJECT_DATA_BY_CLASS]

    for dv in dvs:
        stats_generator.write_stats(dv)
        plotter.plot_data_simple(dv)


if __name__ == '__main__':
    main()
