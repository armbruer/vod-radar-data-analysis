from typing import Dict, List, Tuple
import pandas as pd
from scipy import stats
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

from extraction.file_manager import DataView

"""
EstimatorCollection is intended for working with a collection 
of KernelDensityEstimators.
"""
class EstimatorCollection:
    
    """
    Creates an EstimatorCollection for the given DataVariant and DataViewType.
    
    For each column one estimator is created.
    """
    def __init__(self, data_view: DataView):
        self.data_view = data_view
        self.dfs: List[pd.DataFrame] = self.data_view.df
        self.estimators: List[List[KernelDensityEstimator]] = []
        
        if not isinstance(self.dfs, list):
            self.dfs = [self.dfs]
        
        for i, df in enumerate(self.dfs):
            self.estimators.append([])
            
            for feature in df.columns:
                self.estimators[i].append(KernelDensityEstimator(df=self.data_view, feature=feature))
    
    # no overloading in python, dont need it anysways
    # def __init__(self, 
    #              data_variant: DataVariant, 
    #              data_view_type: DataViewType,
    #              data_manager: DataManager,
    #              data_view: DataView) -> None:
    #     self.data_manager = data_manager
    #     self.data_view = data_view
    #     if self.data_view is None:
    #         self.data_view = data_manager.get_view(data_variant=data_variant, data_view_type=data_view_type)
            
      
    """
    Returns the best bandwidth, kernel found for each estimator (per feature) and for each dataframe
    
    Returns a dictionary of format feature:[bandwith,kernel] per dataframe. 
    """
    def get_hyper_params(self) -> List[Dict[str, Tuple[float, str]]]:
        bws = []
        for kdelist in self.estimators:
            df_bws = {}
            for est in kdelist:
                df_bws[est.feature] = est.bw, est.kernel
                
        return bws
            
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
    def __init__(self, df: pd.DataFrame, feature: str, bw=None, kernel=None) -> None:
        self.df = df
        self.data: np.ndarray = np.atleast_2d(df[feature].to_numpy()).T
        self.bw = bw
        self.kernel = kernel
        self.feature = feature
        if self.bw is None or self.kernel is None:
            self._find_best_estimator()
        
        self.kde = KernelDensity(kernel=self.kernel, bandwidth=self.bw).fit(self.data)
        

    """
    Select best bandwidth using cross-validation.

    See e.g. here https://aakinshin.net/posts/kde-bw/ and 
    https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html .

    We use cross validation from sklearn.
    """
    def _find_best_estimator(self):
        
        # this line is directly copied from seaborn
        # just throw this into bandwiths, maybe its a good default for our data
        # let's see...
        basic_bandwidth = stats.gaussian_kde(self.data.T).scotts_factor() * self.data.std(ddof=1)
        
        
        #bandwidths = np.linspace(1e-3, 1, 9)
        
        bandwidths = np.array([1])
        bandwidths = np.append(bandwidths, basic_bandwidth)
        grid = GridSearchCV(KernelDensity(),
                            {'bandwidth': bandwidths, 
                             'kernel': ['gaussian']}, # , 'cosine', 'epanechnikov'
                            # we restrict kernel to gaussian, as seaborn only plots that
                            cv=5, n_jobs=1, verbose=3)
        grid.fit(self.data)
        self.bw = grid.best_params_['bandwidth']
        self.kernel = grid.best_params_['kernel']


    """
    Samples from the KDE.
    
    :param n_samples: the number of samples to obtain
    
    Returns the samples and scores of the samples in shape (n_samples,)
    """
    def sample(self, n_samples=1)-> Tuple[np.ndarray, np.ndarray]:
        samples = self.kde.sample(n_samples)
        scores = self.kde.score_samples(samples)
        return samples, scores

