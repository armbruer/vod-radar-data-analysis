import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from sklearn.neighbors import KernelDensity

matplotlib.use('Agg') # do not show figures when saving plot
import sys
import os

sys.path.append(os.path.abspath("../view-of-delft-dataset"))


from tqdm import tqdm
from vod.configuration.file_locations import KittiLocations
from extraction.file_manager import DataManager
from extraction.helpers import DataVariant
from extraction.stats_table import StatsTableGenerator
from typing import List, Union
from enum import Enum
from datetime import datetime

class PlotType(Enum):
    VIOLIN = 1,
    BOXPLOT = 2,
    HISTOGRAM = 3,


class ParameterRangePlotter:

    def __init__(self, data_manager: DataManager) -> None:
        self.data_manager = data_manager
        self.kitti_locations = data_manager.kitti_locations

    def plot_data_simple(self, data_variant: DataVariant) -> None:
        plot_types = [PlotType.BOXPLOT, PlotType.VIOLIN, PlotType.HISTOGRAM]

        df = self.data_manager.get_data(data_variant)
        columns = data_variant.column_names()

        self.plot_data(dfs=df, plot_types=plot_types,
                       data_variant=data_variant, value_labels=columns)

    def plot_data(self,
                  dfs: Union[List[pd.DataFrame], pd.DataFrame],
                  plot_types: List[PlotType],
                  data_variant: DataVariant,
                  **kwargs) -> None:
        if not isinstance(dfs, list):
            dfs = [dfs]
        
        cols = len(dfs[0].columns)
        figure_name = kwargs.get('figure_name', data_variant.name.lower())
        value_labels = kwargs.get('value_labels', cols * [''])

        if (not (len(value_labels) == cols)):
            raise ValueError(
                f'Expecting the length of value_labels to be equal to {cols}')

        for k, df in enumerate(dfs):
            index_name = data_variant.index_to_str(k)
            pts = len(plot_types)

            figure, axs = plt.subplots(cols, pts, figsize=(6.4, 10), layout='constrained')

            iter = enumerate(zip(df.items(), value_labels))
            for i, ((_, content), value_label) in tqdm(iter, desc="Preparing subplots"):
                for j, pt in tqdm(enumerate(plot_types), desc="Going through plot types"):

                    if pts > 1 and cols > 1:
                        axis = axs[i, j]
                    elif pts == 1 and cols == 1:
                        axis = axs
                    elif pts == 1:
                        axis = axs[j]  # the other index
                    else:
                        axis = axs[i]  # the other index

                    if pt == PlotType.VIOLIN:
                        gfg = sns.violinplot(y=content, ax=axis)
                        if value_label:
                            gfg.set(ylabel=value_label)
                    elif pt == PlotType.BOXPLOT:
                        gfg = sns.boxplot(y=content, ax=axis)
                        if value_label:
                            gfg.set(ylabel=value_label)
                    elif pt == PlotType.HISTOGRAM:
                        gfg = sns.histplot(x=content, ax=axis, bins=30)
                        if value_label:
                            gfg.set(xlabel=value_label)


            self._store_figure(figure, data_variant, figure_name, index_name, )

    def plot_kneeplot_for_syntactic_data(self) -> None:
        dv = DataVariant.SYNTACTIC_RAD
        rad_df = self.data_manager.get_data()
        doppler_df = rad_df[[2]]
        
        fig, axs = plt.subplots()
        indices = pd.Series(np.arange(0, len(doppler_df), 1), name="Index")
        sns.lineplot(data=doppler_df, x=indices, y='doppler (m/s)')
        axs.grid()
        
        self._store_figure(fig, dv, 'kneeplot')
            
    
    def plot_kde_for_rad(self):
        x_plots = [np.linspace(0, 55, 1000)[:, np.newaxis],
                   np.linspace(-90, 90, 1000)[:, np.newaxis],
                   np.linspace(-25, 25, 1000)[:, np.newaxis]]
        
        self.plot_kde_for_each_parameter(DataVariant.SEMANTIC_RAD, x_plots)
        
        x_plots = [np.linspace(0, 105, 1000)[:, np.newaxis],
                   np.linspace(-180, 180, 1000)[:, np.newaxis],
                   np.linspace(-25, 25, 1000)[:, np.newaxis]]
        
        self.plot_kde_for_each_parameter(DataVariant.SYNTACTIC_RAD, x_plots)
        
    def plot_kde_for_each_parameter(self, df: pd.DataFrame, data_variant: DataVariant, x_plots):
        for (label, content), x_plot in zip(df.items(), x_plots):
            
            colors = ["navy"]
            kernels = ["gaussian"]
            
            fig, ax = plt.subplots()
            
            ax.hist(content, density=True, bins=30) # alpha=0.3
            
            for color, kernel in zip(colors, kernels):
                kde = KernelDensity(kernel=kernel, bandwidth=0.5).fit(X=content)
                log_dens = kde.score_samples(x_plot)
                ax.plot(
                    x_plot[:, 0],
                    np.exp(log_dens), # exp as we get the log-likelihood above
                    color=color,
                    lw=1.5,
                    linestyle="-",
                    label=f"kernel = '{kernel}'",
                )
                
            self._store_figure(fig, data_variant, label.split()[0]) 
        
    def plot_combined(self):
        syntactic_rad_df = self.data_manager.get_data(DataVariant.SYNTACTIC_RAD)
        semantic_rad_df = self.data_manager.get_data(DataVariant.SEMANTIC_RAD)
    
        columns: List[str] = syntactic_rad_df.columns.to_list()
        xlims = [(0, 55), (-90, 90), (-25, 25)]
        
        
        fig, ax = plt.subplots(1, 3, figsize=(10, 4), layout='constrained')

        
        iter = enumerate(zip(syntactic_rad_df, semantic_rad_df, columns, xlims))
        
        for i, (syn_param, sem_param, column, xlim) in iter:
            df_syntactic_rad = pd.DataFrame(data = syntactic_rad_df[syn_param], columns=[column]).assign(annotated = 'No')
            df_semantic_rad = pd.DataFrame(data = semantic_rad_df[sem_param], columns=[column]).assign(annotated = 'Yes')
            df = pd.concat([df_syntactic_rad, df_semantic_rad])
            
            g = sns.histplot(data=df, x=column, hue='annotated', bins=30, ax=ax[i], multiple="dodge", stat="probability", common_norm=False)
            g.set(xlim=xlim)
        
        self._store_figure(fig, figure_name='combined_plot')
        
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
    
    #plotter.plot_combined()
    
    #plotter.plot_kde_for_rad()
    
    stats_generator = StatsTableGenerator(data_manager=dm)

    dvs = [DataVariant.SEMANTIC_RAD, DataVariant.SYNTACTIC_RAD, DataVariant.STATIC_DYNAMIC_RAD,
           DataVariant.SEMANTIC_OBJECT_DATA, DataVariant.SEMANTIC_OBJECT_DATA_BY_CLASS]

    for dv in dvs:
        stats_generator.write_stats(dv)
        plotter.plot_data_simple(dv)


if __name__ == '__main__':
    main()
