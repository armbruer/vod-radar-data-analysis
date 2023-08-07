import logging
import os
from matplotlib import pyplot as plt
import pandas as pd

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from extraction.extimators import KernelDensityEstimator
from extraction.extract import ParameterRangeExtractor
from extraction.helpers import DataVariant, DataViewType
from vod.configuration.file_locations import KittiLocations

class DataView:
    
    def __init__(self, 
                 df: Union[pd.DataFrame, List[pd.DataFrame]], 
                 data_variant: DataVariant, 
                 data_view_type: DataViewType,
                 hyper_params: Dict[str, Dict[str, Any]],
                 kde_columns: List[str]) -> None:
        self.df = df
        self.variant = data_variant
        self.view = data_view_type
        self.hyper_params = hyper_params
        self.kde_columns = kde_columns
        
        self._init_view()
        
    def _init_view(self):
        self._tmp_df: List[pd.DataFrame] = self.df
        if not isinstance(self._tmp_df, list):
            self._tmp_df = [self.df]
            
        columns_to_drop = [col for col in self.view.columns_to_drop() if col in self._tmp_df[0].columns]
        drop_indexes = [self._tmp_df[0].columns.get_loc(col) for col in columns_to_drop]
        self._tmp_df = [df.drop(self.view.columns_to_drop(), axis=1, errors='ignore') for df in self._tmp_df]
        
        self.ticklabels = [label for i, label in enumerate(self.variant.ticklabels()) if i not in drop_indexes]
        self.lims = [lim for i, lim in enumerate(self.variant.lims()) if i not in drop_indexes]
        self.df = self._tmp_df[0] if len(self._tmp_df) == 1 else self._tmp_df
        
        self.kde: Dict[str, KernelDensityEstimator] = {}
        subvariants = self.variant.subvariant_names() if self.variant.subvariant_names() else [None]
        for df, subvariant in zip(self._tmp_df, subvariants):
            for column in self.kde_columns:
                subvariant = f'-{subvariant}' if subvariant is not None else ''
                identifier = f'{self.variant.shortname()}:{column}{subvariant}'
                hyper_params = self.hyper_params[identifier]
                
                self.kde[identifier] = KernelDensityEstimator(df, column, hyper_params['bw'], hyper_params['kernel'])
        
        

class DataManager:
        
    def __init__(self, kitti_locations: KittiLocations, kde_columns: DataViewType = DataViewType.RADE) -> None:
        self.kitti_locations = kitti_locations
        self.extractor = ParameterRangeExtractor(kitti_locations)
        self.data: Dict[DataVariant, Union[List[pd.DataFrame], pd.DataFrame]] = {}
        self.hyperparams: Dict[str, Dict[str, Any]] = {}
        self.data_dir = f'{self.kitti_locations.data_dir}'
        self.hyper_param_dir = f'{self.kitti_locations.hyperparameters_dir}'
        self.kde_columns = kde_columns
        
    def get_view(self, data_variant: DataVariant, 
                 data_view_type: DataViewType = DataViewType.NONE, 
                 refresh=False, 
                 frame_numbers: Optional[List[str]]=None) -> DataView:
        """
        Gets the dataframe for the given data variant either by loading it from an HDF-5 file or by extracting it from the dataset.

        :param data_variant: the data variant for which the dataframe is to be retrieved
        :param data_view: the data view to apply to the data variant, this removes columns unneeded in the current context

        Returns the dataframe containing the data requested
        """
        df = self._get_df(data_variant, refresh, frame_numbers)
        
        subvariants = data_variant.subvariant_names() if data_variant.subvariant_names() else [None]
        
        kde_columns = self.kde_columns.remaining_columns(data_variant)
        
        dfs = [df] if not isinstance(df, list) else df
            
        relevant_hyper_params_only = {}
        for df, subvariant in zip(dfs, subvariants):

            for column in kde_columns:
                subvariant = f'-{subvariant}' if subvariant is not None else ''
                identifier = f'{data_variant.shortname()}:{column}{subvariant}'
                
                relevant_hyper_params_only[identifier] = self.hyperparams[identifier]
        
        return DataView(df=df, data_variant=data_variant, data_view_type=data_view_type, hyper_params=relevant_hyper_params_only, kde_columns=kde_columns)
        
    def _get_df(self, 
                data_variant: DataVariant,
                refresh=False, 
                frame_numbers: Optional[List[str]]=None) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Gets the dataframe for the given data variant either by loading it from an HDF-5 file or by extracting it from the dataset

        :param data_variant: the data variant for which the dataframe is to be retrieved

        Returns the dataframe containing the data requested through the data variant
        """
        if not refresh and (self.data.get(data_variant) is not None 
                            or self.load_dataframe(data_variant) is not None):
            
            dfs = self.data[data_variant]
            if not isinstance(dfs, list):
                dfs = [dfs]
            
            self.load_all_hyperparameters(dfs, data_variant)
            return self.data[data_variant]

        # data extraction takes place here
        if data_variant == DataVariant.SYNTACTIC_DATA:
            df = self.extractor.extract_data_from_syntactic_data(frame_numbers=frame_numbers)
            self.store_dataframe(data_variant=data_variant, df=df)

        elif data_variant == DataVariant.SEMANTIC_DATA:
            df = self.extractor.extract_object_data_from_semantic_data(frame_numbers=frame_numbers)
            self.store_dataframe(data_variant=data_variant, df=df)

        elif data_variant == DataVariant.SEMANTIC_DATA_BY_CLASS:
            semantic_df = self._get_df(DataVariant.SEMANTIC_DATA)
            semantic_by_class = self.extractor.split_by_class(semantic_df)
            self.data[data_variant] = semantic_by_class

        elif data_variant == DataVariant.SYNTACTIC_DATA_BY_MOVING:
            syntactic_df = self._get_df(DataVariant.SYNTACTIC_DATA)
            syntactic_by_moving = self.extractor.split_rad_by_threshold(syntactic_df)
            self.data[data_variant] = syntactic_by_moving
            
        if data_variant in DataVariant.split_variants():
            dfs = self.data[data_variant]
            if self.load_all_hyperparameters(dfs, data_variant):
               return dfs
        else:
            dfs = [self.data[data_variant]]
        
        # after extraction we want to determine the hyperparameters for kde for each feature
        # and store the hyperparameters, as this is expensive operation
        self.store_all_hyperparameters(data_variant, dfs)   
        return self.data[data_variant]


    def load_dataframe(self, data_variant: DataVariant) -> Optional[pd.DataFrame]:
        """
        Loads a dataframe from the most recently saved HDF5-file for this data variant.

        :param data_variant: the data variant of the dataframe to be loaded
        """
        dv_str = data_variant.shortname()

        matching_files = []
        os.makedirs(self.data_dir, exist_ok=True)
        for file in os.listdir(self.data_dir):
            if file.endswith('.hdf5') and dv_str in file:
                datetime_str = file.split('-')[-1].split('.')[0]
                matching_files.append((file, datetime_str))

        matching_files = sorted(
            matching_files, key=lambda x: datetime.strptime(x[1], '%Y_%m_%d_%H_%M_%S'))

        if not matching_files:
            return None

        most_recent: str = matching_files[-1][0]
        
        df = pd.read_hdf(f'{self.data_dir}/{most_recent}', key=dv_str)

        self.data[data_variant] = df
        return df

    def store_dataframe(self, data_variant: DataVariant, df: pd.DataFrame):
        """
        Stores the dataframe in an HDF-5 file using the data variant in the file path.

        :param data_variant: the data_variant of the data to be stored
        :param data: the dataframe to be stored
        """
        if isinstance(df, list):
            raise ValueError('df must not be of type list')

        dv_str = data_variant.shortname()
        os.makedirs(self.data_dir, exist_ok=True)

        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.data[data_variant] = df
        path = f'{self.data_dir}/{dv_str}-{now}.hdf5'
        df.to_hdf(path, key=dv_str, mode='w')
        logging.info(f'Data saved in "file:///{path}"')
        
    def store_hyperparameters(self, hyper_params: Dict[str, Any], data_variant: DataVariant, feature: str, subvariant: Optional[str] = None):
        os.makedirs(self.hyper_param_dir, exist_ok=True)
        
        dv_str = data_variant.shortname()
        subvariant = f'-{subvariant}' if subvariant is not None else ''
        
        hyp_df = pd.DataFrame(hyper_params)
        path = f'{self.hyper_param_dir}/{dv_str}/{feature}{subvariant}.csv'
        
        hyp_df.to_csv(path, index=False)
        
        identifier = f'{dv_str}:{feature}{subvariant}'
        self.hyperparams[identifier] = hyper_params
        logging.info(f'Stored hyperparameters in "file:///{path}"')
    
    def store_all_hyperparameters(self, data_variant, dfs):
        if not isinstance(dfs, list):
            dfs = [dfs]
        
        subvariants = data_variant.subvariant_names() if data_variant.subvariant_names() else [None]
        for df, subvariant in zip(dfs, subvariants):
            columns = self.kde_columns.remaining_columns(data_variant)
            for column in columns:
                kde = KernelDensityEstimator(df, column)
                hyper_params = {'bw': kde.bw, 'kernel': kde.kernel}
                
                self.store_hyperparameters(hyper_params=hyper_params, data_variant=data_variant, feature=column, subvariant=subvariant)
    

    def load_all_hyperparameters(self, dfs: List[pd.DataFrame], data_variant: DataVariant) -> bool:
        subvariants = data_variant.subvariant_names() if data_variant.subvariant_names() else [None]
        hyps_missing: bool = True
        
        for df, subvariant in zip(dfs, subvariants):
            for column in df.columns:
                hyper_params = self.load_hyperparameters(data_variant=data_variant, feature=column, subvariant=subvariant)
                if hyper_params is None:
                    hyps_missing = False
                    
        return hyps_missing
                
    def load_hyperparameters(self, data_variant: DataVariant, feature: str, subvariant: Optional[str] = None) -> Optional[Dict[str, Any]]:
        dv_str = data_variant.shortname()
        subvariant = f'-{subvariant}' if subvariant is not None else ''
        identifier = f'{dv_str}:{feature}{subvariant}'
        
        if identifier in self.hyperparams:
            return self.hyperparams[identifier]
        
        os.makedirs(self.hyper_param_dir, exist_ok=True)
        path = f'{self.hyper_param_dir}/{dv_str}/{feature}{subvariant}.csv'
        
        if not os.path.exists(path):
            return None
        
        hyp_df = pd.read_csv(path)
        # we expect to have only one value per column
        # this is generic enough for our needs to store kde hyper parameters
        hyper_params = {k: v[0] for k,v in hyp_df.to_dict(orient='list')}
        
        self.hyperparams[identifier] = hyper_params
        return hyper_params
    