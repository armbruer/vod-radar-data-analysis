import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.neighbors import KernelDensity

matplotlib.use('Agg') # do not show figures when saving plot
import sys
import os

sys.path.append(os.path.abspath("../view-of-delft-dataset"))


from tqdm import tqdm
from vod.configuration.file_locations import KittiLocations
from extraction.file_manager import DataManager
from extraction.stats_table import StatsTableGenerator
from extraction.extract import DataVariant, ParameterRangeExtractor
from typing import List
from enum import Enum
from datetime import datetime

class PlotType(Enum):
    VIOLIN = 1,
    BOXPLOT = 2,
    HISTOGRAM = 3,
    KNEEPLOT = 4


class ParameterRangePlotter:

    def __init__(self, data_manager: DataManager) -> None:
        self.data_manager = data_manager
        self.kitti_locations = data_manager.kitti_locations
        self.extractor = ParameterRangeExtractor(self.data_manager)

    def plot_data_simple(self, data_variant: DataVariant) -> None:
        plot_types = [PlotType.BOXPLOT, PlotType.VIOLIN, PlotType.HISTOGRAM]

        data = self.extractor.get_data(data_variant)
        columns = data_variant.column_names(with_unit=True)

        self.plot_data(data=data, plot_types=plot_types,
                       data_variant=data_variant, value_labels=columns)

    def plot_data(self,
                  data: List[np.ndarray],
                  plot_types: List[PlotType],
                  data_variant: DataVariant,
                  **kwargs) -> None:
        
        cols = data[0].shape[1] if data[0].ndim > 1 else 1
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
                    param: np.ndarray = d[:, i] if d.ndim == 2 else d[:]
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
                        gfg.set(xlabel=value_label)
                    elif pt == PlotType.KNEEPLOT:
                        indices = np.arange(0, param.shape[0], 1)
                        gfg = sns.lineplot(x=indices, y=param)
                        axis.grid()
                        gfg.set(xlabel=other_label, ylabel=value_label)

            # plt.show()

            self._store_figure(figure, data_variant, figure_name, index_name, )

    def _store_figure(self, figure, data_variant=None, figure_name='', index_name='', subdir=''):
        figures_dir = f"{self.kitti_locations.figures_dir}"
        if data_variant is not None:
            figures_dir = f"{figures_dir}/{data_variant.name.lower()}"
        elif subdir:
            figures_dir = f"{figures_dir}/{subdir}"
            
        os.makedirs(figures_dir, exist_ok=True)

        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        path = f"{figures_dir}/{figure_name}_{index_name}_{now}"
        figure.savefig(f'{path}.png', format='png')
        figure.savefig(f'{path}.pdf', format='pdf')
        logging.info(f'Plot generated in file:///{path}.png')

    def plot_kneeplot(self, data: np.ndarray, **kwargs) -> None:
        if data.ndim != 1:
            raise ValueError(
                'Expecting each parameter distribution to be of dimension 1')

        self.plot_data([np.sort(data)], [PlotType.KNEEPLOT], DataVariant.SYNTACTIC_RAD, **kwargs)

    def plot_kneeplot_from_syntactic_data(self) -> None:
        rad = self.extractor.get_data(DataVariant.SYNTACTIC_RAD)

        kwargs = {
            'value_labels': ['doppler (m/s)'],
            'other_labels': ['index'],
            'figure_name':  'kneeplot',
        }
        self.plot_kneeplot(data=rad[0][:, 2], **kwargs)
    
    def plot_kde_for_each_parameter(self, data_variant: DataVariant, x_plots):
        data = self.extractor.get_data(data_variant)
        columns = data_variant.column_names()
        
        for param, column, x_plot in zip(data[0].T, columns, x_plots):
            
            colors = ["navy"]
            kernels = ["gaussian"]
            
            fig, ax = plt.subplots()
            
            ax.hist(param, density=True, bins=30, alpha=0.3)
            
            for color, kernel in zip(colors, kernels):
                kde = KernelDensity(kernel=kernel, bandwidth=0.5).fit(X=param.reshape(param.shape[0], 1))
                log_dens = kde.score_samples(x_plot)
                ax.plot(
                    x_plot[:, 0],
                    np.exp(log_dens), # exp as we get the log-likelihood above
                    color=color,
                    lw=1.5,
                    linestyle="-",
                    label=f"kernel = '{kernel}'",
                )
                
            self._store_figure(fig, data_variant, column.lower())
            
    
    def plot_kde_for_rad(self):
        x_plots = [np.linspace(0, 55, 1000)[:, np.newaxis],
                   np.linspace(-90, 90, 1000)[:, np.newaxis],
                   np.linspace(-25, 25, 1000)[:, np.newaxis]]
        
        self.plot_kde_for_each_parameter(DataVariant.SEMANTIC_RAD, x_plots)
        
        x_plots = [np.linspace(0, 105, 1000)[:, np.newaxis],
                   np.linspace(-180, 180, 1000)[:, np.newaxis],
                   np.linspace(-25, 25, 1000)[:, np.newaxis]]
        
        self.plot_kde_for_each_parameter(DataVariant.SYNTACTIC_RAD, x_plots)
        
        
    def plot_combined(self):
        syntactic_rad = self.extractor.get_data(DataVariant.SYNTACTIC_RAD)
        semantic_rad = self.extractor.get_data(DataVariant.SEMANTIC_RAD)
    
        columns = DataVariant.SEMANTIC_RAD.column_names(with_unit=True)
        xlims = [(0, 55), (-90, 90), (-25, 25)]
        
        
        fig, ax = plt.subplots(1, 3, figsize=(10, 4), layout='constrained')

        
        iter = enumerate(zip(syntactic_rad[0].T, semantic_rad[0].T, columns, xlims))
        
        for i, (syn_param, sem_param, column, xlim) in iter:
            g = sns.histplot([syn_param, sem_param], bins=30, color=['r', 'b'], ax=ax[i], multiple="dodge")
            
            ax[i].legend(loc='upper right', labels=['unannotated', 'annotated'])
            g.set(xlabel=column)
            g.set(xlim=xlim)
            g.set_yscale('log')
        
        self._store_figure(fig, figure_name='combined_plot')

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

    dm = DataManager(kitti_locations=kitti_locations)
    plotter = ParameterRangePlotter(data_manager=dm)
    
    plotter.plot_combined()
    
    #plotter.plot_kde_for_rad()
    
    # stats_generator = StatsTableGenerator(data_manager=dm)

    # dvs = [DataVariant.SEMANTIC_RAD, DataVariant.SYNTACTIC_RAD, DataVariant.STATIC_DYNAMIC_RAD,
    #        DataVariant.SEMANTIC_OBJECT_DATA, DataVariant.SEMANTIC_OBJECT_DATA_BY_CLASS]

    # for dv in dvs:
    #     stats_generator.write_stats(dv)
    #     plotter.plot_data_simple(dv)
    
    # #plotter.plot_kneeplot_from_syntactic_data()


if __name__ == '__main__':
    main()
