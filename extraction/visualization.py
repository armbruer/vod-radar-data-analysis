# TODO
import sys
import os
sys.path.append(os.path.abspath("../view-of-delft-dataset"))

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from datetime import datetime
from enum import Enum
from typing import List
from extraction import DataVariant, ParameterRangeExtractor
from extraction.stats_table import StatsTableGenerator
from vod.configuration.file_locations import KittiLocations



class PlotType(Enum):
    VIOLIN = 1,
    BOXPLOT = 2,
    HISTOGRAM = 3,
    KNEEPLOT = 4


class ParameterRangePlotter:

    def __init__(self, kitti_locations: KittiLocations) -> None:
        self.kitti_locations = kitti_locations

    def plot_parameters(self,
                        parameters: List[np.ndarray],
                        plot_types: List[PlotType],
                        data_variant: DataVariant = None, 
                        **kwargs) -> None:
        
        figure_name = kwargs.get('figure_name', 'parameters')
        figure_title = kwargs.get('figure_title')
        value_labels = kwargs.get('value_labels', len(parameters) * [''])
        other_labels = kwargs.get('other_labels', len(parameters) * [''])
        
        if(not(len(value_labels) == len(other_labels) == len(parameters))):
            raise ValueError('Expecting the length of value_labels, other_labels and parameters to be equal')
        
        if plot_types[0] != PlotType.KNEEPLOT:
            for p in parameters:
                if p.ndim != 1:
                    raise ValueError(
                        'Expecting each parameter distribution to be of dimension 1')

        figure, axs = plt.subplots(len(plot_types), len(value_labels))

        for i, value_label in enumerate(value_labels):
            for j, pt in enumerate(plot_types):
                param = parameters[i]
                other_label = other_labels[i]
                
                if len(plot_types) > 1 and len(value_labels) > 1:
                    axis = axs[i, j]
                elif len(plot_types) == 1 and len(value_labels) == 1:
                    axis = axs
                elif len(plot_types) == 1:
                    axis = axs[j] # the other index
                else:
                    axis = axs[i] # the other index

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
                

        figure.tight_layout()
        #plt.show()
        if figure_title:
            figure.suptitle(figure_title)
        
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        path = f"{self.kitti_locations.output_dir}/{data_variant.name.lower() if data_variant is not None else figure_name}_{now}"
        figure.savefig(f'{path}.svg', format='svg')
        figure.savefig(f'{path}.png', format='png')

    @staticmethod
    def get_title_from_data_variant(data_variant: DataVariant) -> str:
        if data_variant == DataVariant.DYNAMIC_RAD:
            return 'Range, azimuth, doppler for dynamic, syntactic data'
        elif data_variant == DataVariant.SEMANTIC_RAD:
            return 'Range, azimuth, doppler for semantic data'
        elif data_variant == DataVariant.STATIC_RAD:
            return 'Range, angle, doppler for static, syntactic data'
        elif data_variant == DataVariant.SYNTACTIC_RAD:
            return 'Range, angle, doppler for syntactic data'
        
        raise ValueError('Unknown data_variant')

    def plot_kneeplot(self, param: np.ndarray, **kwargs) -> None:
        if param.ndim != 1:
            raise ValueError(
                'Expecting each parameter distribution to be of dimension 1')

        self.plot_parameters([np.sort(param)], [PlotType.KNEEPLOT], **kwargs)

    def plot_rad_data(self, data_variant: DataVariant) -> None:
        plot_types = [PlotType.BOXPLOT, PlotType.VIOLIN, PlotType.HISTOGRAM]
        extractor = ParameterRangeExtractor(self.kitti_locations)
        
        rad = list(extractor.get_data(data_variant).T)
        self.plot_parameters(parameters=rad, 
                             plot_types=plot_types, 
                             data_variant=data_variant, 
                             value_labels=ParameterRangeExtractor.names_rad_with_unit(), 
                             figure_title=ParameterRangePlotter.get_title_from_data_variant(data_variant=data_variant))
        

        
    def plot_kneeplot_from_syntactic_data(self) -> None:
        extractor = ParameterRangeExtractor(self.kitti_locations)
        rad = extractor.get_data(DataVariant.SYNTACTIC_RAD)
        
        kwargs = {
            'value_labels': ['doppler (m/s)'],
            'other_labels': ['index'],
            'figure_name':  'kneeplot',
            'figure_title': 'kneeplot doppler (syntactic data)',
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

    abs = lambda p: os.path.abspath(p)
    print(f"Radar directory: {abs(kitti_locations.radar_dir)}")
    print(f"Label directory: {abs(kitti_locations.label_dir)}")
    print(f"Output directory: {abs(kitti_locations.output_dir)}")

    plotter = ParameterRangePlotter(kitti_locations=kitti_locations)
    stats_generator = StatsTableGenerator(kitti_locations=kitti_locations)
    stats_generator.write_stats(DataVariant.SYNTACTIC_RAD)

    #plotter.plot_kneeplot_from_syntactic_data()
    #plotter.plot_rad_data(DataVariant.SYNTACTIC_RAD)



if __name__ == '__main__':
    main()
