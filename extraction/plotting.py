import matplotlib
from matplotlib.image import AxesImage
matplotlib.use('Agg') # disable interactive matplotlib backend
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os

from tqdm import tqdm
from typing import Dict, List, Union
from enum import Enum
from datetime import datetime
from itertools import product

from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from extraction.estimators import KernelDensityEstimator
from extraction.file_manager import DataManager, DataView
from extraction.helpers import DataVariant, DataViewType, get_class_id_from_name, get_class_ids, get_class_names, get_name_from_class_id

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

    def plot_data_simple(self, 
                         plot_types: List[PlotType], 
                         data_variants: List[DataVariant] = DataVariant.all_variants(),
                         data_view_type: DataViewType = DataViewType.EASY_PLOTABLE) -> None:
        
        for dv in data_variants:
            data_view: DataView = self.data_manager.get_view(dv, data_view_type)
            self.plot_data(dfs=data_view.df, plot_types=plot_types, data_variant=dv)
            
    def plot_data_simple_improved(self, 
                                  plot_types: List[PlotType]=[PlotType.HISTOGRAM], 
                                  data_variants: List[DataVariant] = [DataVariant.SEMANTIC_DATA],
                                  data_view_type: DataViewType = DataViewType.EASY_PLOTABLE) -> None:
        
        for dv in data_variants:
            data_view: DataView = self.data_manager.get_view(dv, data_view_type)
            self.plot_data_improved(data_view=data_view, plot_types=plot_types, data_variant=dv, figure_name='improved')

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
        
    def plot_data_improved(self,
                  data_view: DataView,
                  plot_types: List[PlotType],
                  data_variant: DataVariant,
                  figure_name='') -> None:
        dfs = data_view.df
        if not isinstance(dfs, list):
            dfs = [dfs]
        
        plot_functions = []
        
        x_pts = [PlotType.HISTOGRAM, PlotType.HIST_KDE, PlotType.KDE]
        
        # TODO this needs bandwiths for kde
        if PlotType.HISTOGRAM in plot_types:
            plot_functions.append((PlotType.HISTOGRAM, lambda g: g.map_dataframe(sns.histplot, x="value", bins=30, stat="probability")))
            plot_functions.append((PlotType.HISTOGRAM, lambda g: g.map_dataframe(sns.histplot, x="value", bins=30, stat="count", log_scale=(False, True))))
        if PlotType.HIST_KDE in plot_types:   
            plot_functions.append((PlotType.HIST_KDE, lambda g: g.map_dataframe(sns.histplot, x="value", bins=30, stat="density", kde=True)))
        if PlotType.KDE in plot_types:
            plot_functions.append((PlotType.KDE, lambda g: g.map_dataframe(sns.kdeplot, x="value")))
        if PlotType.VIOLIN in plot_types:
            plot_functions.append((PlotType.VIOLIN, lambda g: g.map_dataframe(sns.violinplot, y="value")))
        if PlotType.BOXPLOT in plot_types:
            plot_functions.append((PlotType.BOXPLOT, lambda g: g.map_dataframe(sns.boxplot, y="value")))
            
            
        for k, df_orig in enumerate(dfs):
            index_name = data_variant.index_to_str(k)
            
            df: pd.DataFrame = pd.melt(df_orig, value_vars=df_orig.columns, var_name='param')
            for j, (plot_type, pf) in enumerate(plot_functions):
                
                g = sns.FacetGrid(df, col_wrap=4, height=2.5, aspect=1, col='param', legend_out=True, sharex=False, sharey=False)
                
                pf(g)
                g.set_titles("{col_name}")
                #g.tight_layout()

                # for i, ax in enumerate(g.axes.flat):
                #     if plot_type in x_pts:
                #         g.set_xlabels("")
                #         ax.set_xlim(data_view.lims[i])
                #         if data_view.ticklabels[i] is not None:
                #             ax.set_xticklabels(data_view.ticklabels[i])
                #     else:
                #         g.set_ylabels("")
                #         ax.set_ylim(data_view.lims[i])
                #         if data_view.ticklabels[i] is not None:
                #             ax.set_ytickslabels(data_view.ticklabels[i])
                
                plot_name = plot_type.name.lower()
                fig_name = f'{data_variant.shortname()}-{figure_name}-{plot_name}-{j}'
                
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
                ('hist', lambda ax, df, column: sns.histplot(data=df, x=column, bins=30, ax=ax, stat="probability")),
                ('hist_kde', lambda ax, df, column: sns.histplot(data=df, x=column, bins=30, ax=ax, stat="density", kde=True)),
                ('kde', lambda ax, df, column: sns.kdeplot(data=df, x=column, ax=ax))
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
                
                for i in range(nrows):
                    for j in range(ncols):
                        param, column, xlim = next(iter)
                        data = rad_df[param]
                        df = pd.DataFrame(data=data, columns=[column])
                        
                        if column == 'Elevation [degree]':
                            # this is a bit of a lazy hack, we know almost all data except for outliers
                            # falls into this xlim (from previous rounds of running this) range, 
                            # so just throw away the rest of the data since it is anyways not visualized
                            # the advantage here is that we will get 30 bins which are actually visualized
                            # this is only okay and needed because the outliers are spread accross so many degrees
                            # if we don't do this we get only 3-4 bars in the area where we have 95% of the data
                            df = self._droplims(df, xlim, column)
                    
                        axis = ax[i, j] if data_view_type == DataViewType.RADE else ax[j]
                        
                        g = pf(axis, df, column)
                        g.set(xlim=xlim)
            
                self._store_figure(fig, figure_name=f'{dv.shortname()}-{dvt_str}-{fig_name}', subdir=f'{dvt_str}')
        
    def plot_by_class_combined(self, most_important_only: bool = False):
        data_view: DataView = self.data_manager.get_view(DataVariant.SEMANTIC_DATA_BY_CLASS, DataViewType.RADE)
        object_class_dfs = data_view.df
        
        indexes = get_class_ids()
        if most_important_only:
            # only keep the most important classes
            indexes = [get_class_id_from_name(name, summarized=True) for name in ['car', 'pedestrian', 'cyclist']]
            object_class_dfs = [object_class_dfs[i] for i in indexes]
        
        columns: List[str] = object_class_dfs[0].columns.to_list()
        by_column_dfs = self._map_to_single_class_column_dfs(object_class_dfs, columns, indexes)
        
        plot_functions = [
            ('kde', lambda i, j, df, column: sns.kdeplot(data=df, x=column, hue='clazz', ax=ax[i, j], common_norm=False)),
            ('hist_step', lambda i, j, df, column: sns.histplot(data=df, x=column, hue='clazz', bins=30, ax=ax[i, j], element="step", stat="probability", common_norm=False))
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
                    
                    if column == 'Elevation [degree]':
                        # this is a bit of a lazy hack, we know almost all data except for outliers
                        # falls into this xlim (from previous rounds of running this) range, 
                        # so just throw away the rest of the data since it is anyways not visualized
                        # the advantage here is that we will get 30 bins which are actually visualized
                        # this is only okay and needed because the outliers are spread accross so many degrees
                        # if we don't do this we get only 3-4 bars in the area where we have 95% of the data
                        df = self._droplims(df, xlim, column)
                    
                    
                    g = pf(i, j, df, column)
                    g.set(xlim=xlim)
                    
                    if not most_important_only:
                        plt.setp(g.get_legend().get_texts(), fontsize='8') 
                        plt.setp(g.get_legend().get_title(), fontsize='9', text="Class")
                    else: 
                        plt.setp(g.get_legend().get_title(), text="Class")
                    #sns.move_legend(g, loc=1, bbox_to_anchor=(1, 1))
        
            self._store_figure(fig, figure_name=f'{name}-{fig_name}', subdir=f'{name}')
            
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
            ('hist_step_kde', lambda i, j, df, column: sns.histplot(data=df, x=column, hue='annotated', bins=30, ax=ax[i, j], element="step", stat="density", common_norm=False, kde=True)),
            ('kde', lambda i, j, df, column: sns.kdeplot(data=df, x=column, hue='annotated', ax=ax[i, j], common_norm=False))
        ]
        
        for fig_name, pf in plot_functions:
            fig, ax = plt.subplots(2, 2, figsize=(8, 6), layout='constrained')
            
            iter = zip(syntactic_rad_df, semantic_rad_df, columns, semantic_dv.lims)
            for i in range(2):
                for j in range(2):
                    syn_param, sem_param, column, xlim = next(iter)
                    syn_data = syntactic_rad_df[syn_param]
                    sem_data = semantic_rad_df[sem_param]
                    
                    df_syntactic_rad = pd.DataFrame(data = syn_data, columns=[column]).assign(annotated = 'No')
                    df_semantic_rad = pd.DataFrame(data = sem_data, columns=[column]).assign(annotated = 'Yes')
                    df = pd.concat([df_syntactic_rad, df_semantic_rad])
                    # we need to cut all data to the same x-ranges so we can actually compare
                    # the syntactic data to the semantic data
                    # alternatively we could also leave everything uncut but then we would have to show all outliers
                    df = self._droplims(df, xlim, column)
                    
                    g = pf(i, j, df, column)
                    g.set(xlim=xlim)
            
            self._store_figure(fig, figure_name=f'syn_sem_combined-{fig_name}', subdir='syn_sem_combined')
        
    def plot_azimuth_heatmap(self):
        semantic_dfs = self.data_manager.get_view(data_variant=DataVariant.SEMANTIC_DATA_BY_CLASS, 
                                                  data_view_type=DataViewType.PLOT_LONG_LAT)
        
        for df, clazz in zip(semantic_dfs.df, get_class_names()):
            fig, ax = plt.subplots()
            
            df = df.astype(int)

            # just add a 0 everywhere
            all_xy = {(0, x, y) 
                      for x, y in product(range(-26, 27), range(0, 53))}
            
            df_extend = pd.DataFrame(all_xy, columns=df.columns)
            df = pd.concat([df, df_extend], ignore_index=True)
            
            # remove outliers (don't need'em for this plot)
            # remember columns are weirdly named for this because of the camera coordinate system
            df["y"] = df["y"].apply(lambda y: -y)
            df.drop(df[(df.y < -26) | (df.y > 26) | (df.x < 0) | (df.x > 52)].index, inplace=True)
            df = df.pivot_table(index="x", columns="y", values="Detections [#]", aggfunc=np.sum)
            df = df.fillna(0)
            
            ax = sns.heatmap(df, norm=LogNorm(), cbar=True, cmap=sns.cm._cmap_r, ax=ax, vmin=0, square=True)
            ax.set_title(f'{clazz.capitalize()}s')
            ax.invert_yaxis()
            ax.set_ylim((0, 52))
            ax.set_xlim((0, 53))
            ax.set_xticks([6, 16, 26, 36, 46], labels=[-20, -10, 0, 10, 20], rotation=0)
            ax.set_yticks([0, 10, 20, 30, 40, 50], labels=[0, 10, 20, 30, 40, 50], rotation=0)
            ax.set_xlabel("Lat. Distance [m]")
            ax.set_ylabel("Long. Distance [m]")
            ax.set_facecolor('#23275b')
            
            self._store_figure(fig, figure_name=clazz, subdir='azi_heatmaps')
            
        
    def plot_ele_heatmap(self):
        semantic_dfs = self.data_manager.get_view(data_variant=DataVariant.SEMANTIC_DATA_BY_CLASS, 
                                                  data_view_type=DataViewType.PLOT_ALT_LONG)
        
        for df, clazz in zip(semantic_dfs.df, get_class_names()):
            fig, ax = plt.subplots()
            
            df = df.astype(int)
            
            # just add a 0 everywhere
            all_xz = {(0, x, z) 
                       for x, z in product(range(0, 53), range(-6, 7))}
            
            df_extend = pd.DataFrame(all_xz, columns=df.columns)
            df = pd.concat([df, df_extend], ignore_index=True)
            
            # remove outliers (don't need'em for this plot)
            
            df.drop(df[(df.x < 0) | (df.x > 52) | (df.z < -6) | (df.z > 6)].index, inplace=True)
            df = df.pivot_table(index="z", columns="x", values="Detections [#]", aggfunc=np.sum)
            df = df.fillna(0)
            
            ax = sns.heatmap(df, norm=LogNorm(), cbar=True, 
                             cmap=sns.cm._cmap_r, ax=ax, vmin=0, square=True,
                             cbar_kws = dict(use_gridspec=False,location="top"))
            
            ax.set_title(f'{clazz.capitalize()}s')
            ax.invert_yaxis()
            ax.set_ylim((0, 12))
            ax.set_xlim((0, 52))
            ax.set_yticks([1, 6, 11], labels=[-5, 0, 5], rotation=0)
            ax.set_xticks([0, 10, 20, 30, 40, 50], labels=[0, 10, 20, 30, 40, 50], rotation=0)
            ax.set_xlabel("Long. Distance [m]")
            ax.set_ylabel("Alt. Distance [m]")
            ax.set_facecolor('#23275b')
            
            self._store_figure(fig, figure_name=clazz, subdir='ele_heatmaps')
    
            
    def plot_relationships(self, data_variants: List[DataVariant] = DataVariant.all_variants()):
        for data_variant in data_variants:
            data_view: DataView = self.data_manager.get_view(data_variant=data_variant, 
                                                            data_view_type=DataViewType.EASY_PLOTABLE)
            
            dfs = data_view.df
            if not isinstance(dfs, list):
                dfs = [dfs]
                
            for df in dfs:
                fig, ax = plt.subplots()
                sns.pairplot(data=df)
                figure_name= f'relationships-{data_view.variant.shortname()}-{data_view.view.name.lower()}'
                self._store_figure(fig, figure_name=figure_name, subdir='relationships')
            
    def correlation_heatmap(self, data_variant: DataVariant):
        data_view: DataView = self.data_manager.get_view(data_variant=data_variant, 
                                                         data_view_type=DataViewType.CORR_HEATMAP)
        dfs = data_view.df
        if not isinstance(dfs, list):
            dfs = [dfs]
            
        for df in dfs:
            corr = df.select_dtypes('number').corr()
            fig, ax = plt.subplots()
            ax = sns.heatmap(corr, ax=ax, cbar=True, annot=True, annot_kws={"fontsize":8})
            figure_name= f'corr-heatmap-{data_view.variant.shortname()}-{data_view.view.name.lower()}'
            self._store_figure(fig, figure_name=figure_name, subdir='corr-heatmaps')
        
        
    def _store_figure(self, 
                      figure: Union[Figure, sns.FacetGrid], 
                      data_variant: DataVariant=None, 
                      figure_name: str='', 
                      index_name: str='', 
                      subdir: str='', 
                      timestring: bool=False):
        
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
        logging.info(f'Plot generated in "file:///{path}.pdf"')
        # don't forget closing the figure, otherwise matplotlib likes to keep'em in RAM :)
        if isinstance(figure, Figure): # can also be a FacetGrid
            plt.close(figure)
    
    def _droplims(self, df, lims, column):
        xmin, xmax = lims
        return df.drop(df[(df[column] < xmin) | (df[column] > xmax)].index)
    
    def plot_all_kdeplots(self, data_variants: List[DataVariant] = DataVariant.all_variants(), 
                          data_view_type: DataViewType = DataViewType.RADE):
        
        for data_variant in data_variants:
            data_view = self.data_manager.get_view(data_variant=data_variant, 
                                                data_view_type=data_view_type, no_hyperparams=False)
            
            self.kde: Dict[str, KernelDensityEstimator] = {}
            subvariants = data_variant.subvariant_names() if data_variant.subvariant_names() else [None]
            for df in zip(self._tmp_df, subvariants):
                for column in data_view.kde_columns:
                    subvariant = f'-{subvariant}' if subvariant is not None else ''
                    self.kdeplot(df, data_view, self.kde, column, subvariant)
                    self.kdeplot(df, data_view, self.kde, column, subvariant, show_underlying_data=True)
        
    
    def kdeplot(self, 
                df: pd.DataFrame, 
                data_view: DataView, 
                kde: KernelDensityEstimator, 
                feature: str,
                subvariant: str = '', 
                show_underlying_data: bool = False):
        # code adapted from: https://scikit-learn.org/stable/auto_examples/neighbors/plot_kde_1d.html

        with sns.plotting_context(context='paper'):
            with sns.axes_style(style='ticks'):
                sns.set_palette(sns.color_palette())

                # prepare plot data
                xmin, xmax = data_view.lims
                X_plot = np.linspace(xmin, xmax)[:, np.newaxis]
                
                dens = np.exp(kde.kde.score_samples(X_plot))

                fig, ax = plt.subplots()
                
                if show_underlying_data:
                    # plot a stepwise histogram
                    ax.hist(df[feature], ec='0.3', fc='orange', histtype='step', density=True, alpha=0.2)
                
                ax.plot(
                    X_plot[:, 0],
                    dens,
                    color='orange',
                    linestyle="-",
                )                    
                ax.legend(loc="upper right")

                ax.set_xlim(xmin, xmax)
                ax.set_ylim(0, 1.0)
                
                #ax.set_title(feature)
                #ax.set_xticks(data_view.ticklabels, labels=data_view.ticklabels, rotation=0)
                ax.set_xlabel(feature)
                ax.set_ylabel("Density")
                
                hist = 'hist' if show_underlying_data else ''
                fig_name = f'{feature}-{hist}'
                
                self._store_figure(fig,
                                   figure_name=fig_name,
                                   subdir='kdeplots',
                                   index_name=subvariant,
                                   data_variant=data_view.variant,
                                   timestring=True)