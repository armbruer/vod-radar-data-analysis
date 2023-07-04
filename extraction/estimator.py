from typing import List, Tuple
from numpy import ndarray
import pandas as pd
from scipy import stats
import numpy as np
from sklearn.neighbors import KernelDensity

from extraction.file_manager import DataManager, DataView
from extraction.helpers import DataVariant, DataViewType

"""
EstimatorCollection is intended for working with a collection 
of estimators.
"""
class EstimatorCollection:
    
    """
    Creates an EstimatorCollection for the given DataVariant and DataViewType.
    
    For each column one estimator is created.
    """
    def __init__(self, data_manager: DataManager, data_variant: DataVariant, data_view_type: DataViewType) -> None:
        self.data_manager = data_manager
        self.data_view = data_manager.get_view(data_variant=data_variant, data_view_type=data_view_type)
        self.dfs: List[pd.DataFrame] = self.data_view.df
        self.estimators: List[List[KernelDensityEstimator]] = []
        
        if not isinstance(self.dfs, list):
            self.dfs = [self.dfs]
        
        for i, df in enumerate(self.dfs):
            self.estimators.append([])
            
            for feature in df.columns:
                self.estimators[i].append(KernelDensityEstimator(data_view=self.data_view, feature=feature))
            
    """
    Draws n_samples from each estimator in the collection.
    
    :param n_samples: the number of samples to draw from each estimator in the collection
    :param subvariant_index: the index of the current subvariant 
    (only relevant for DataVariant.SEMANTIC_BY_CLASS AND DataVariant.SYNTACTIC_BY_MOVING)
    
    Returns a tuple containing a list of samples and a list of scores. 
    Each entry has the shape (n_samples,). 
    """
    def draw_sample(self, n_samples=1, subvariant_index=0) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        samples_list: List[np.ndarray] = []
        scores_list: List[np.ndarray] = []
        
        for estimator in self.estimators[subvariant_index]:
            samples, scores = estimator.sample(n_samples)          
            samples_list.append(samples)
            scores_list.append(scores)
            
        return samples_list, scores_list
            

"""
Creates a KDE for a feature that can be sampled from.
"""
class KernelDensityEstimator:
    
    """
    Creates a KDE.
    
    :param data_view: the data_view to use the data from
    :param feature: the feature of the data_view to create the estimate for
    """
    def __init__(self, data_view: DataView, feature: str, bw=None) -> None:
        self.data_view = data_view
        self.data: ndarray = data_view.df[feature].to_numpy()
        self.bw = bw
        if self.bw is None:
            # this line is directly copied from seaborn, we want to have the same bandwidth as in the plots!  
            self.bw = stats.gaussian_kde(self.data.T).scotts_factor() * self.data.std(ddof=1)
        
        self.kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(self.data)

    """
    Samples from the KDE.
    
    :param n_samples: the number of samples to obtain
    
    Returns the samples and scores of the samples in shape (n_samples,)
    """
    def sample(self, n_samples=1)-> Tuple[np.ndarray, np.ndarray]:
        samples = self.kde.sample(n_samples)
        scores = self.kde.score_samples(samples)
        return samples, scores


