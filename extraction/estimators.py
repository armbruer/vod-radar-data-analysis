import pandas as pd
import numpy as np

from typing import List, Tuple
from scipy import stats
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

"""
EstimatorCollection is intended for working with a collection 
of KernelDensityEstimators.
"""
class EstimatorCollection:
    
    """
    Creates an EstimatorCollection for the given DataVariant and DataViewType.
    
    For each column one estimator is created.
    """
    def __init__(self, dfs: List[pd.DataFrame]):
        self.dfs = dfs
        self.estimators: List[List[KernelDensityEstimator]] = []
        
        if not isinstance(self.dfs, list):
            self.dfs = [self.dfs]
        
        for i, df in enumerate(self.dfs):
            self.estimators.append([])
            
            for feature in df.columns:
                self.estimators[i].append(KernelDensityEstimator(df=self.data_view, feature=feature))
    
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
            self._estimate_best_hyperparameters()
        
        self.kde = KernelDensity(kernel=self.kernel, bandwidth=self.bw).fit(self.data)
        

    """
    Select best bandwidth using cross-validation.

    See e.g. here https://aakinshin.net/posts/kde-bw/ and 
    https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html .

    We use cross validation from sklearn.
    """
    def _estimate_best_hyperparameters(self):
        
        # this line is directly copied from seaborn
        # just throw this into bandwiths, maybe its a good default for our data
        # let's see...
        basic_bandwidth = stats.gaussian_kde(self.data.T).scotts_factor() * self.data.std(ddof=1)
        
        print(f"Data size: {self.data.size}")
        bandwidths = np.linspace(0.25, 1.5, 9)
        bandwidths = np.append(bandwidths, basic_bandwidth)
        # höhere bandwidth
        # 
        
        grid = GridSearchCV(KernelDensity(algorithm='kd_tree', kernel='epanechnikov'),
                            {'bandwidth': bandwidths},
                            # 'kernel': ['gaussian', 'cosine', 'epanechnikov']}, 
                            # we restrict ourselves to epanechnikov, as it is fast
                            # higher jobs number does not seem to improve the situation
                            # use kdtree, as we have 1 dim data
                            # https://scikit-learn.org/stable/faq.html#why-do-i-sometime-get-a-crash-freeze-with-n-jobs-1-under-osx-or-linux
                            cv=3, n_jobs=8, verbose=3)
        grid.fit(self.data)
        self.bw = grid.best_params_['bandwidth']
        self.kernel = 'epanechnikov'
        print(f'Optimal bandwidth: {self.bw}')
        
        # grid_results = grid.cv_results_
        # plt.plot(bandwidths, grid_results['mean_test_score'])
        # plt.xlabel('Bandwidth')
        # plt.ylabel('Cross-validation accuracy')
        # plt.show()


    """
    Samples from the KDE.
    
    :param n_samples: the number of samples to obtain
    
    Returns the samples and scores of the samples in shape (n_samples,)
    """
    def sample(self, n_samples=1)-> Tuple[np.ndarray, np.ndarray]:
        samples = self.kde.sample(n_samples)
        scores = self.kde.score_samples(samples)
        return samples, scores