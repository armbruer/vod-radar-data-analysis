import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import os

matplotlib.use('Agg') # do not show figures when saving plot
from tqdm import tqdm
from extraction.file_manager import DataManager
from extraction.helpers import DataVariant, DataView, get_name_from_class_id
from typing import List, Union
from enum import Enum
from datetime import datetime

class PlotType(Enum):
    VIOLIN = 1,
    BOXPLOT = 2,
    HISTOGRAM = 3

class ParameterRangePlotter:

    def __init__(self, data_manager: DataManager) -> None:
        self.data_manager = data_manager
        self.kitti_locations = data_manager.kitti_locations

    def plot_data_simple(self, data_variant: DataVariant) -> None:
        plot_types = [PlotType.BOXPLOT, PlotType.VIOLIN, PlotType.HISTOGRAM]

        df = self.data_manager.get_df(data_variant, DataView.PLOTABLE)

        self.plot_data(dfs=df, plot_types=plot_types, data_variant=data_variant)

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
        dv = DataVariant.SYNTACTIC_DATA
        rad_df = self.data_manager.get_df(dv, DataView.RAD)
        doppler_df = rad_df[[2]]
        
        fig, axs = plt.subplots()
        indices = pd.Series(np.arange(0, len(doppler_df), 1), name="Index")
        sns.lineplot(data=doppler_df, x=indices, y='doppler (m/s)')
        axs.grid()
        
        self._store_figure(fig, dv, 'kneeplot')
            
    
    def plot_rad(self):
        for dv in DataVariant.basic_variants():
            rad_df = self.data_manager.get_df(DataVariant.SEMANTIC_DATA, DataView.RAD)
        
            columns: List[str] = rad_df.columns.to_list()
            xlims = [(0, 55), (-90, 90), (-25, 25)]


            plot_functions = [
                ('hist', lambda i: sns.histplot(data=df, x=column, bins=30, ax=ax[i], stat="probability")),
                ('hist_kde', lambda i: sns.histplot(data=df, x=column, bins=30, ax=ax[i], stat="probability", kde=True)),
                ('kde', lambda i: sns.kdeplot(data=df, x=column, ax=ax[i]))
            ]
            
            for fig_name, pf in plot_functions:
                fig, ax = plt.subplots(1, 3, figsize=(10, 4), layout='constrained')
                iter = enumerate(zip(rad_df, columns, xlims))
                
                for i, (param, column, xlim) in iter:
                    df = pd.DataFrame(data = rad_df[param], columns=[column])
                    
                    g = pf(i)
                        
                    g.set(xlim=xlim)
            
                self._store_figure(fig, figure_name=f'{dv}-rad-{fig_name}')
        
        
    def plot_by_class_combined(self, kde: bool = False):
        object_class_dfs = self.data_manager.get_df(DataVariant.SEMANTIC_DATA_BY_CLASS, DataView.RAD)
    
        columns: List[str] = object_class_dfs[0].columns.to_list()
        xlims = [(0, 55), (-90, 90), (-25, 25)]
        
        by_column_dfs: List[List[pd.DataFrame]] = [[], [], []]
        for i, c in enumerate(columns):
            for class_id, df in enumerate(object_class_dfs):
                by_column_dfs[i].append(df[[c]].assign(clazz = get_name_from_class_id(class_id)))
                                        
        by_column_dfs = list(map(pd.concat, by_column_dfs))
        
        iter = enumerate(zip(by_column_dfs, columns, xlims))
        fig, ax = plt.subplots(1, 3, figsize=(10, 4), layout='constrained')
        for i, (df, column, xlim) in iter:
             
            if kde:
                g = sns.kdeplot(data=df, x=column, hue='clazz', ax=ax[i], common_norm=False)
            else:
                g = sns.histplot(data=df, x=column, hue='clazz', bins=30, ax=ax[i], multiple="layer", stat="probability", common_norm=False)
            g.set(xlim=xlim)
            #sns.move_legend(g, loc=1, bbox_to_anchor=(1, 1))
            
        self._store_figure(fig, figure_name='classes_combined_plot')
        
    def plot_syn_sem_combined(self, kde: bool = False):
        syntactic_rad_df = self.data_manager.get_df(DataVariant.SYNTACTIC_DATA, DataView.RAD)
        semantic_rad_df = self.data_manager.get_df(DataVariant.SEMANTIC_DATA, DataView.RAD)
    
        columns: List[str] = syntactic_rad_df.columns.to_list()
        xlims = [(0, 55), (-90, 90), (-25, 25)]
        
        
        fig, ax = plt.subplots(1, 3, figsize=(10, 4), layout='constrained')

        
        iter = enumerate(zip(syntactic_rad_df, semantic_rad_df, columns, xlims))
        
        for i, (syn_param, sem_param, column, xlim) in iter:
            df_syntactic_rad = pd.DataFrame(data = syntactic_rad_df[syn_param], columns=[column]).assign(annotated = 'No')
            df_semantic_rad = pd.DataFrame(data = semantic_rad_df[sem_param], columns=[column]).assign(annotated = 'Yes')
            df = pd.concat([df_syntactic_rad, df_semantic_rad])
            
            if kde:
                g = sns.kdeplot(data=df, x=column, hue='annotated', ax=ax[i], common_norm=False)
            else:
                g = sns.histplot(data=df, x=column, hue='annotated', bins=30, ax=ax[i], multiple="dodge", stat="probability", common_norm=False)
            g.set(xlim=xlim)
        
        self._store_figure(fig, figure_name='syn_sem_combined_plot')
        
    def _store_figure(self, figure, data_variant: DataVariant =None, figure_name: str ='', index_name: str ='', subdir: str=''):
        figures_dir = f"{self.kitti_locations.figures_dir}"
        if data_variant is not None:
            figures_dir = f"{figures_dir}/{data_variant.name.lower()}"
        elif subdir:
            figures_dir = f"{figures_dir}/{subdir}"
            
        os.makedirs(figures_dir, exist_ok=True)

        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        path = f"{figures_dir}/{figure_name}_{index_name}_{now}"
        #figure.savefig(f'{path}.png', format='png')
        figure.savefig(f'{path}.pdf', format='pdf')
        logging.info(f'Plot generated in file:///{path}.pdf')

def run_basic_visualization(plotter : ParameterRangePlotter):
    for dv in DataVariant.all_variants():
        plotter.plot_data_simple(dv)

