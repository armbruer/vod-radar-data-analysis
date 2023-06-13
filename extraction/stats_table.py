# TODO
import sys
import os
sys.path.append(os.path.abspath("../view-of-delft-dataset"))


from extraction import DataVariant, ParameterRangeExtractor
import numpy as np
import pandas as pd
from datetime import datetime

from vod.configuration.file_locations import KittiLocations

class StatsTableGenerator:
    
    def __init__(self, kitti_locations: KittiLocations) -> None:
        self.kitti_locations = kitti_locations
        

    def write_stats(self, data_variant: DataVariant) -> None:
        ex = ParameterRangeExtractor(kitti_locations=self.kitti_locations)
        data: np.ndarray = ex.get_data(data_variant=data_variant)
        
        if data.ndim != 2:
            raise ValueError('Dimension of retrieved data must be two')
        
        mins = np.min(data, axis=0)
        maxs = np.max(data, axis=0)
        means = np.mean(data, axis=0)
        vars = np.mean(data, axis=0)
        stds = np.std(data, axis=0)
        
        stats = np.vstack((mins, maxs, means, vars, stds))
        columns = DataVariant.column_names_with_unit()
        columns = list(map(lambda c: c.capitalize(), columns))
        
        df = pd.DataFrame(stats, columns=columns)
        df.insert(0, "Name", pd.Series(["Min", "Max", "Mean", "Var", "Std"]))
        
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        
        outpath = f'{self.kitti_locations.output_dir}/{data_variant.name.lower()}-{now}'
        df.to_csv(f'{outpath}.csv', index=False)
        # Requires latex installation with booktabs
        # TODO decimal separator
        df.to_latex(
            f'{outpath}.tex',
            float_format="%.2f",
            label=f"table:{data_variant.name.lower()}", 
            position="htb!",
            column_format=len(columns) * "c", 
            index=False,
        )

