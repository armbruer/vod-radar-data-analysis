from tqdm import tqdm
from typing import List, Union
from enum import Enum
from datetime import datetime
from itertools import product
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from extraction.file_manager import DataManager
from extraction.helpers import DataVariant, DataView, get_class_id_from_name, get_class_ids, get_class_names, get_name_from_class_id

import matplotlib
matplotlib.use('Agg') # do not show figures when saving plot
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os

class PlotType(Enum):
    VIOLIN = 1,
    BOXPLOT = 2,
    HISTOGRAM = 3

class ParameterRangePlotter:

    def __init__(self, data_manager: DataManager) -> None:
        self.data_manager = data_manager
        self.kitti_locations = data_manager.kitti_locations
    
    # for debugging
    def plot_xyz(self):
        df = self.data_manager.get_df(DataVariant.SEMANTIC_DATA_BY_CLASS, DataView.PLOT_XYZ_ONLY)
        self.plot_data(dfs=df, plot_types=[PlotType.HISTOGRAM], data_variant=DataVariant.SEMANTIC_DATA_BY_CLASS)

    def plot_data_simple(self, plot_types: List[PlotType]) -> None:
        for dv in DataVariant.all_variants():
            df = self.data_manager.get_df(dv, DataView.EASY_PLOTABLE)
            self.plot_data(dfs=df, plot_types=plot_types, data_variant=dv)

    def plot_data(self,
                  dfs: Union[List[pd.DataFrame], pd.DataFrame],
                  plot_types: List[PlotType],
                  data_variant: DataVariant,
                  **kwargs) -> None:
        if not isinstance(dfs, list):
            dfs = [dfs]
        
        cols = len(dfs[0].columns)
        figure_name = kwargs.get('figure_name', data_variant.shortname())
        value_labels = kwargs.get('value_labels', cols * [''])

        if (not (len(value_labels) == cols)):
            raise ValueError(
                f'Expecting the length of value_labels to be equal to {cols}')

        for k, df in enumerate(dfs):
            index_name = data_variant.index_to_str(k)
            pts = len(plot_types)

            figure, axs = plt.subplots(cols, pts, figsize=(3*pts, 2*cols), layout='constrained')

            iter = enumerate(zip(df.items(), value_labels))
            for i, ((_, content), value_label) in tqdm(iter, desc="Preparing subplots"):
                for j, pt in tqdm(enumerate(plot_types), desc="Going through plot types"):

                    if pts > 1 and cols > 1:
                        axis = axs[i, j]
                    elif pts == 1 and cols == 1:
                        axis = axs
                    elif pts == 1:
                        axis = axs[j+i*pts]  # the other index
                    else:
                        axis = axs[i+j*pts]  # the other index

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
                            gfg.set_yscale('log')


            self._store_figure(figure, data_variant, figure_name, index_name, )

    def plot_kneeplot_for_syntactic_data(self) -> None:
        dv = DataVariant.SYNTACTIC_DATA
        rad_df = self.data_manager.get_df(dv, DataView.RAD)
        doppler_df = rad_df[[2]]
        
        fig, axs = plt.subplots()
        indices = pd.Series(np.arange(0, len(doppler_df), 1), name="Index")
        sns.lineplot(data=doppler_df, x=indices, y='Doppler [m/s]')
        axs.grid()
        
        self._store_figure(fig, dv, 'kneeplot')
        
        
    def plot_rade(self):
        for dv in DataVariant.basic_variants():
            rad_df = self.data_manager.get_df(dv, DataView.RADE)
        
            columns: List[str] = rad_df.columns.to_list()
            xlims = [(0, 55), (-90, 90), (-25, 25), (-90, 90)]


            plot_functions = [
                ('hist', lambda i, j, df, column: sns.histplot(data=df, x=column, bins=30, ax=ax[i, j], stat="probability")),
                ('hist_kde', lambda i, j, df, column: sns.histplot(data=df, x=column, bins=30, ax=ax[i, j], stat="density", kde=True)),
                ('kde', lambda i, j, df, column: sns.kdeplot(data=df, x=column, ax=ax[i, j]))
            ]
            
            for fig_name, pf in plot_functions:
                fig, ax = plt.subplots(2, 2, figsize=(8, 6), layout='constrained')
                iter = zip(rad_df, columns, xlims)
                
                for i in range(2):
                    for j in range(2):
                        param, column, xlim = next(iter)
                        df = pd.DataFrame(data = rad_df[param], columns=[column])
                    
                        g = pf(i, j, df, column)
                        
                        g.set(xlim=xlim)
            
                self._store_figure(fig, figure_name=f'{dv.shortname()}-rade-{fig_name}', subdir='rade')
    
    def plot_rad(self):
        for dv in DataVariant.basic_variants():
            rad_df = self.data_manager.get_df(dv, DataView.RAD)
        
            columns: List[str] = rad_df.columns.to_list()
            xlims = [(0, 55), (-90, 90), (-25, 25)]


            plot_functions = [
                ('hist', lambda i, df, column: sns.histplot(data=df, x=column, bins=30, ax=ax[i], stat="probability")),
                ('hist_kde', lambda i, df, column: sns.histplot(data=df, x=column, bins=30, ax=ax[i], stat="density", kde=True)),
                ('kde', lambda i, df, column: sns.kdeplot(data=df, x=column, ax=ax[i]))
            ]
            
            for fig_name, pf in plot_functions:
                fig, ax = plt.subplots(1, 3, figsize=(10, 3), layout='constrained')
                iter = enumerate(zip(rad_df, columns, xlims))
                
                for i, (param, column, xlim) in iter:                        
                    df = pd.DataFrame(data = rad_df[param], columns=[column])
                
                    g = pf(i, df, column)
                    g.set(xlim=xlim)
            
                self._store_figure(fig, figure_name=f'{dv.shortname()}-rad-{fig_name}', subdir='rad')
        
        
    def plot_by_class_combined(self):
        object_class_dfs = self.data_manager.get_df(DataVariant.SEMANTIC_DATA_BY_CLASS, DataView.RADE)
    
        columns: List[str] = object_class_dfs[0].columns.to_list()
        xlims = [(0, 55), (-90, 90), (-25, 25), (-90, 90)]
        
        by_column_dfs = self._map_to_single_class_column_dfs(object_class_dfs, columns, get_class_ids())
        
        plot_functions = [
            ('kde', lambda i, j, df, column: sns.kdeplot(data=df, x=column, hue='clazz', ax=ax[i, j], common_norm=False))
        ]
        
        # hack: haven't found the proper way to access these properties :(
        # so set them globally and then reset them afterwards
        mpl.rcParams['legend.labelspacing'] = 0.2
        mpl.rcParams['legend.handlelength'] = 1.0
        
        for fig_name, pf in plot_functions:
            fig, ax = plt.subplots(2, 2, figsize=(8, 8), layout='constrained')
            iter = zip(by_column_dfs, columns, xlims)
            
            for i in range(2):
                for j in range(2):
                    
                    df, column, xlim = next(iter)
                    
                    g = pf(i, j, df, column)
                    g.set(xlim=xlim)
                    
                    plt.setp(g.get_legend().get_texts(), fontsize='8') 
                    plt.setp(g.get_legend().get_title(), fontsize='9', text="Class")
                    #sns.move_legend(g, loc=1, bbox_to_anchor=(1, 1))
        
            self._store_figure(fig, figure_name=f'classes-rade-{fig_name}', subdir='classes-rade')
            
        # reset to default
        mpl.rcParams['legend.labelspacing'] = 0.5
        mpl.rcParams['legend.handlelength'] = 2.0
        
        
    def plot_by_class_combined_main_only(self):
        object_class_dfs = self.data_manager.get_df(DataVariant.SEMANTIC_DATA_BY_CLASS, DataView.RADE)
        # only keep the main classes
        indexes = [get_class_id_from_name(name, summarized=True) for name in ['car', 'pedestrian', 'cyclist']]
        object_class_dfs = [object_class_dfs[i] for i in indexes]
        
        columns: List[str] = object_class_dfs[0].columns.to_list()
        # TODO different xlims for syntactic data
        xlims = [(0, 55), (-90, 90), (-25, 25), (-90, 90)]
        
        by_column_dfs = self._map_to_single_class_column_dfs(object_class_dfs, columns, indexes)
        
        plot_functions = [
            ('kde', lambda i, j, df, column: sns.kdeplot(data=df, x=column, hue='clazz', ax=ax[i, j], common_norm=False))
        ]
        
        for fig_name, pf in plot_functions:
            fig, ax = plt.subplots(2, 2, figsize=(8, 8), layout='constrained')
            iter = zip(by_column_dfs, columns, xlims)
            
            for i in range(2):
                for j in range(2):
                    
                    df, column, xlim = next(iter)
                    
                    g = pf(i, j, df, column)
                    g.set(xlim=xlim)
                    plt.setp(g.get_legend().get_title(), text="Class")
                    #sns.move_legend(g, loc=1, bbox_to_anchor=(1, 1))
        
            self._store_figure(fig, figure_name=f'main-classes-rade-{fig_name}', subdir='main_classes_rade')

    def _map_to_single_class_column_dfs(self, object_class_dfs, columns, class_ids):
        by_column_dfs: List[List[pd.DataFrame]] = [[], [], [], []]
        for i, c in enumerate(columns):
            for class_id, df in zip(class_ids, object_class_dfs):
                by_column_dfs[i].append(df[[c]].assign(clazz = get_name_from_class_id(class_id, summarized=True)))
                                        
        by_column_dfs = list(map(pd.concat, by_column_dfs))
        return by_column_dfs
        
    def plot_syn_sem_combined(self):
        syntactic_rad_df = self.data_manager.get_df(DataVariant.SYNTACTIC_DATA, DataView.RADE)
        semantic_rad_df = self.data_manager.get_df(DataVariant.SEMANTIC_DATA, DataView.RADE)
    
        columns: List[str] = syntactic_rad_df.columns.to_list()
        xlims = [(0, 55), (-90, 90), (-25, 25), (-90, 90)]
        
        
        plot_functions = [
            ('hist', lambda i, j, df, column: sns.histplot(data=df, x=column, hue='annotated', bins=30, ax=ax[i, j], multiple="dodge", stat="probability", common_norm=False)),
            ('hist_kde', lambda i, j, df, column: sns.histplot(data=df, x=column, hue='annotated', bins=30, ax=ax[i, j], multiple="dodge", stat="density", common_norm=False, kde=True)),
            ('hist_step', lambda i, j, df, column: sns.histplot(data=df, x=column, hue='annotated', bins=30, ax=ax[i, j], element="step", stat="probability", common_norm=False)),
            ('kde', lambda i, j, df, column: sns.kdeplot(data=df, x=column, hue='annotated', ax=ax[i, j], common_norm=False))
        ]
        
        for fig_name, pf in plot_functions:
            fig, ax = plt.subplots(2, 2, figsize=(8, 6), layout='constrained')
            
            iter = zip(syntactic_rad_df, semantic_rad_df, columns, xlims)
            for i in range(2):
                for j in range(2):
                    syn_param, sem_param, column, xlim = next(iter)
            
                    df_syntactic_rad = pd.DataFrame(data = syntactic_rad_df[syn_param], columns=[column]).assign(annotated = 'No')
                    df_semantic_rad = pd.DataFrame(data = semantic_rad_df[sem_param], columns=[column]).assign(annotated = 'Yes')
                    df = pd.concat([df_syntactic_rad, df_semantic_rad])
                    g = pf(i, j, df, column)

                    g.set(xlim=xlim)
            
            self._store_figure(fig, figure_name=f'syn_sem_combined-{fig_name}', subdir='syn_sem_combined')
        
    def plot_heatmap(self):
        semantic_dfs = self.data_manager.get_df(DataVariant.SEMANTIC_DATA_BY_CLASS, DataView.PLOT_LONG_LAT)
        
        for df, clazz in zip(semantic_dfs, get_class_names()):
            fig, ax = plt.subplots()
            
            df = df.round(decimals=0).astype(int)

            # just add a 0 everywhere
            all_xy = {(0, x, y) for x, y in product(range(-25, 26), range(0, 53))}
            # found_xy = set()
            # for _, row in df.iterrows():
            #     value = row['x'], row['y']
            #     found_xy.add(value)
            
            # missing_xy = all_xy - found_xy
            # missing_xy = list(map(lambda x, y: (x, y, 0), missing_xy))
            
            
            df_extend = pd.DataFrame(all_xy, columns=df.columns)
            df = pd.concat([df, df_extend], ignore_index=True)
            
            # remove outliers (don't need'em for this plot)
            # remember columns are weirdly named for this because of the radar coordinate system
            df.drop(df[(df.y < -25) | (df.y > 25) | (df.x < 0) | (df.x > 52)].index, inplace=True)
            df = df.pivot_table(index="x", columns="y", values="Detections [#]", aggfunc=np.sum)
            
              
            ax = sns.heatmap(df, norm=LogNorm(), cbar=True, cmap=sns.cm._cmap_r, ax=ax)
            ax.set_title(f'{clazz.capitalize()}s')
            #print(ax.get_xticks())
            #ax.set_xticks(np.array([-20, 0, 20]))
            # ax.set_xticklabels(np.array([-20, 0, 20]))
            # ax.set_yticks(np.array([0, 20, 40]))
            # ax.set_yticklabels(np.array([0, 20, 40]))
            ax.set_xlabel("Lat. Distance [m]")
            ax.set_ylabel("Long. Distance [m]")
            ax.invert_yaxis()
            ax.set_facecolor('#23275b')
            
            self._store_figure(fig, figure_name=clazz, subdir='heatmaps')
        
        
        
    def _store_figure(self, figure, data_variant: DataVariant=None, figure_name: str='', index_name: str='', subdir: str='', timestring: bool=False):
        figures_dir = f"{self.kitti_locations.figures_dir}"
        if data_variant is not None:
            figures_dir = f"{figures_dir}/{data_variant.shortname()}"
        elif subdir:
            figures_dir = f"{figures_dir}/{subdir}"
            
        os.makedirs(figures_dir, exist_ok=True)
        path = f"{figures_dir}/{figure_name}_{index_name}"
        
        if timestring:
            now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            path = f"{path}/_{now}"
        
        #figure.savefig(f'{path}.png', format='png')
        figure.savefig(f'{path}.pdf', format='pdf', bbox_inches='tight')
        logging.info(f'Plot generated in file:///{path}.pdf')
        # don't forget closing the figure, otherwise matplotlib likes to keep'em in RAM :)
        if isinstance(figure, Figure): # can also be a fa
            plt.close(figure)

