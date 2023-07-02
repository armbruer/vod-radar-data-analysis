from tqdm import tqdm
from typing import List, Union
from enum import Enum
from datetime import datetime
from itertools import product
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from extraction.file_manager import DataManager, DataView
from extraction.helpers import DataVariant, DataViewType, get_class_id_from_name, get_class_ids, get_class_names, get_name_from_class_id

import matplotlib
matplotlib.use('Agg') # disable interactive matplotlib backend
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
    HISTOGRAM = 3,
    KDE = 4,
    HIST_KDE = 5

class DistributionPlotter:

    def __init__(self, data_manager: DataManager) -> None:
        self.data_manager = data_manager
        self.kitti_locations = data_manager.kitti_locations
    
    # for debugging
    def plot_xyz(self):
        data_view: DataView = self.data_manager.get_view(DataVariant.SEMANTIC_DATA_BY_CLASS, DataViewType.PLOT_XYZ_ONLY)
        self.plot_data(dfs=data_view.df, plot_types=[PlotType.HISTOGRAM], data_variant=data_view.variant)

    def plot_data_simple(self, plot_types: List[PlotType]) -> None:
        for dv in DataVariant.all_variants():
            data_view: DataView = self.data_manager.get_view(dv, DataViewType.EASY_PLOTABLE)
            self.plot_data(dfs=data_view.df, plot_types=plot_types, data_variant=dv)
            
    def plot_data_test(self) -> None:
        for dv in [DataVariant.SEMANTIC_DATA]:
            data_view: DataView = self.data_manager.get_view(dv, DataViewType.EASY_PLOTABLE)
            self.plot_data_improved(dfs=data_view.df, plot_types=[PlotType.HISTOGRAM], data_variant=dv, figure_name='test')

    def plot_data(self,
                  dfs: Union[List[pd.DataFrame], pd.DataFrame],
                  plot_types: List[PlotType],
                  data_variant: DataVariant,
                  figure_name=None) -> None:
        if not isinstance(dfs, list):
            dfs = [dfs]
        
        params = len(dfs[0].columns)
        
        if figure_name is None: 
            figure_name = data_variant.shortname()
            

        for k, df in enumerate(dfs):
            index_name = data_variant.index_to_str(k)
            pts = len(plot_types)


            figure, axs = plt.subplots(params, pts, figsize=(3*pts, 2*params), layout='constrained')

            for i, (_, content) in tqdm(enumerate(df.items()), desc="Preparing subplots"):
                for j, pt in tqdm(enumerate(plot_types), desc="Going through plot types"):

                    if pts > 1 and params > 1:
                        axis = axs[i, j]
                    elif pts == 1 and params == 1:
                        axis = axs
                    elif pts == 1:
                        axis = axs[j+i*pts]  # the other index
                    else:
                        axis = axs[i+j*pts]  # the other index

                    if pt == PlotType.VIOLIN:
                        gfg = sns.violinplot(y=content, ax=axis)
                    elif pt == PlotType.BOXPLOT:
                        gfg = sns.boxplot(y=content, ax=axis)
                    elif pt == PlotType.HISTOGRAM:
                        gfg = sns.histplot(x=content, ax=axis, bins=30)
                        gfg.set_yscale('log')


            self._store_figure(figure, data_variant, figure_name, index_name)
        
    # TODO everywhere: log    
        
    def plot_data_improved(self,
                  dfs: Union[List[pd.DataFrame], pd.DataFrame],
                  plot_types: List[PlotType],
                  data_variant: DataVariant,
                  figure_name='') -> None:
        if not isinstance(dfs, list):
            dfs = [dfs]
        
        plot_functions = []
        
        if PlotType.HISTOGRAM in plot_types:
            plot_functions.append(('hist', lambda g: g.map_dataframe(sns.histplot, x="value", bins=30, stat="probability")))
        if PlotType.HIST_KDE in plot_types:   
            plot_functions.append(('hist_kde', lambda g: g.map_dataframe(sns.histplot, x="value", bins=30, stat="density", kde=True)))
        if PlotType.KDE in plot_types:
            plot_functions.append(('kde', lambda g: g.map_dataframe(sns.kdeplot, x="value")))
        if PlotType.VIOLIN in plot_types:
            plot_functions.append(('violin', lambda g: g.map_dataframe(sns.violinplot, y="value")))
        if PlotType.BOXPLOT in plot_types:
            plot_functions.append(('boxplot', lambda g: g.map_dataframe(sns.boxplot, y="value")))
            
            
        for k, df in enumerate(dfs):
            index_name = data_variant.index_to_str(k)
            
            df = pd.melt(df, value_vars=df.columns, var_name='param')
            for plot_name, pf in plot_functions:
                
                g = sns.FacetGrid(df, col_wrap=4, height=2.5, aspect=1, col='param', legend_out=True, sharex=False, sharey=False)
                
                pf(g)
                
                fig_name = f'{data_variant.shortname()}-{figure_name}-{plot_name}'
                self._store_figure(g, data_variant, fig_name, index_name)

    def plot_kneeplot_for_syntactic_data(self) -> None:
        dv = DataVariant.SYNTACTIC_DATA
        data_view: DataView = self.data_manager.get_view(dv, DataViewType.NONE)
        df = data_view.df
        df = df.sort_values(by='Doppler Compensated [m/s]')
        df['Index'] = range(0, len(df))
        kneeplot_df = df[['Index', 'Doppler Compensated [m/s]']]
        
        fig, axs = plt.subplots()

        sns.lineplot(data=kneeplot_df, x='Index', y='Doppler Compensated [m/s]')
        axs.grid()
        
        fig.show() # you need this to zoom in
        
        self._store_figure(fig, dv, 'kneeplot')
        
    def plot_rade(self, 
                  data_variants: List[DataVariant] = DataVariant.basic_variants(), 
                  data_view_type: DataViewType=DataViewType.RADE):
        for dv in data_variants:
            data_view: DataView = self.data_manager.get_view(dv, data_view_type)
            dvt_str = data_view.view.name.lower()
            rad_df = data_view.df
            columns: List[str] = rad_df.columns.to_list()

            plot_functions = [
                ('hist', lambda i, j, df, column: sns.histplot(data=df, x=column, bins=30, ax=ax[i, j], stat="probability")),
                ('hist_kde', lambda i, j, df, column: sns.histplot(data=df, x=column, bins=30, ax=ax[i, j], stat="density", kde=True)),
                ('kde', lambda i, j, df, column: sns.kdeplot(data=df, x=column, ax=ax[i, j]))
            ]
            
            if data_view_type == DataViewType.RADE:
                nrows = 2
                ncols = 2
            elif data_view_type == DataViewType.RAD:
                nrows = 1
                ncols = 3
            else:
                raise ValueError(f'Unexpected value for data_view_type {data_view_type}')
            
            for fig_name, pf in plot_functions:
                fig, ax = plt.subplots(nrows, ncols, figsize=(8, 6), layout='constrained')
                iter = zip(rad_df, columns, data_view.lims)
                
                for i in range(2):
                    for j in range(2):
                        param, column, xlim = next(iter)
                        df = pd.DataFrame(data = rad_df[param], columns=[column])
                    
                        g = pf(i, j, df, column)
                        
                        g.set(xlim=xlim)
            
                self._store_figure(fig, figure_name=f'{dv.shortname()}-{dvt_str}-{fig_name}', subdir=f'{dvt_str}')
        
    def plot_by_class_combined(self, most_important_only: bool = False):
        data_view: DataView = self.data_manager.get_view(DataVariant.SEMANTIC_DATA_BY_CLASS, DataViewType.RADE)
        object_class_dfs = data_view.df
        
        if most_important_only:
            # only keep the most important classes
            indexes = [get_class_id_from_name(name, summarized=True) for name in ['car', 'pedestrian', 'cyclist']]
            object_class_dfs = [object_class_dfs[i] for i in indexes]
        
        columns: List[str] = object_class_dfs[0].columns.to_list()
        by_column_dfs = self._map_to_single_class_column_dfs(object_class_dfs, columns, indexes)
        
        plot_functions = [
            ('kde', lambda i, j, df, column: sns.kdeplot(data=df, x=column, hue='clazz', ax=ax[i, j], common_norm=False))
        ]
        
        if not most_important_only:
            # hack: haven't found the proper way to access these properties :(
            # so set them globally and then reset them afterwards
            mpl.rcParams['legend.labelspacing'] = 0.2
            mpl.rcParams['legend.handlelength'] = 1.0
        
        name = 'classes-rade' if not most_important_only else 'main-classes-rade'
        
        for fig_name, pf in plot_functions:
            fig, ax = plt.subplots(2, 2, figsize=(8, 8), layout='constrained')
            iter = zip(by_column_dfs, columns, data_view.lims)
            
            for i in range(2):
                for j in range(2):
                    
                    df, column, xlim = next(iter)
                    
                    g = pf(i, j, df, column)
                    g.set(xlim=xlim)
                    
                    if not most_important_only:
                        plt.setp(g.get_legend().get_texts(), fontsize='8') 
                        plt.setp(g.get_legend().get_title(), fontsize='9', text="Class")
                    else: 
                        plt.setp(g.get_legend().get_title(), text="Class")
                    #sns.move_legend(g, loc=1, bbox_to_anchor=(1, 1))
        
            self._store_figure(fig, figure_name=f'{name}-{fig_name}', subdir='{name}')
            
        if not most_important_only:
            # reset global stuff to default
            mpl.rcParams['legend.labelspacing'] = 0.5
            mpl.rcParams['legend.handlelength'] = 2.0

    def _map_to_single_class_column_dfs(self, object_class_dfs, columns, class_ids):
        by_column_dfs: List[List[pd.DataFrame]] = [[], [], [], []]
        for i, c in enumerate(columns):
            for class_id, df in zip(class_ids, object_class_dfs):
                by_column_dfs[i].append(df[[c]].assign(clazz = get_name_from_class_id(class_id, summarized=True)))
                                        
        by_column_dfs = list(map(pd.concat, by_column_dfs))
        return by_column_dfs
        
    def plot_syn_sem_combined(self, data_view_type: DataViewType = DataViewType.RADE):
        syntactic_dv: DataView = self.data_manager.get_view(DataVariant.SYNTACTIC_DATA, data_view_type)
        semantic_dv: DataView = self.data_manager.get_view(DataVariant.SEMANTIC_DATA, data_view_type)
    
        syntactic_rad_df = syntactic_dv.df
        semantic_rad_df = semantic_dv.df

        columns: List[str] = syntactic_rad_df.columns.to_list()
        
        
        plot_functions = [
            ('hist', lambda i, j, df, column: sns.histplot(data=df, x=column, hue='annotated', bins=30, ax=ax[i, j], multiple="dodge", stat="probability", common_norm=False)),
            ('hist_kde', lambda i, j, df, column: sns.histplot(data=df, x=column, hue='annotated', bins=30, ax=ax[i, j], multiple="dodge", stat="density", common_norm=False, kde=True)),
            ('hist_step', lambda i, j, df, column: sns.histplot(data=df, x=column, hue='annotated', bins=30, ax=ax[i, j], element="step", stat="probability", common_norm=False)),
            ('kde', lambda i, j, df, column: sns.kdeplot(data=df, x=column, hue='annotated', ax=ax[i, j], common_norm=False))
        ]
        
        for fig_name, pf in plot_functions:
            fig, ax = plt.subplots(2, 2, figsize=(8, 6), layout='constrained')
            
            iter = zip(syntactic_rad_df, semantic_rad_df, columns, semantic_dv.lims)
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
        semantic_dfs = self.data_manager.get_view(DataVariant.SEMANTIC_DATA_BY_CLASS, DataViewType.PLOT_LONG_LAT)
        
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
        
        
        
    def _store_figure(self, figure: Union[Figure, sns.FacetGrid], data_variant: DataVariant=None, figure_name: str='', index_name: str='', subdir: str='', timestring: bool=False):
        figures_dir = f"{self.kitti_locations.figures_dir}"
        if data_variant is not None:
            path = f"{figures_dir}/{data_variant.shortname()}/"
        elif subdir:
            path = f"{figures_dir}/{subdir}/"
            
        os.makedirs(path, exist_ok=True)
        filename = ''
        if figure_name:
            filename = f"{filename}-{figure_name}"
        
        if index_name:
            filename = f"{filename}-{index_name}"
        
        if timestring:
            now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            filename = f"{filename}/-{now}"
            
        if filename.startswith("-"):
            filename = filename[1:]
            
        path = f"{path}{filename}"
        
        #figure.savefig(f'{path}.png', format='png')
        figure.savefig(f'{path}.pdf', format='pdf', bbox_inches='tight')
        logging.info(f'Plot generated in file:///{path}.pdf')
        # don't forget closing the figure, otherwise matplotlib likes to keep'em in RAM :)
        if isinstance(figure, Figure): # can also be a FacetGrid
            plt.close(figure)

