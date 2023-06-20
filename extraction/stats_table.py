from extraction.file_manager import DataManager
from datetime import datetime

import logging
import os
import numpy as np
import pandas as pd

from extraction.helpers import DataVariant

class StatsTableGenerator:

    def __init__(self, data_manager: DataManager) -> None:
        self.data_manager = data_manager
        self.kitti_locations = data_manager.kitti_locations

    def write_stats(self, data_variant: DataVariant) -> None:
        data = self.data_manager.get_df_plot_ready(data_variant=data_variant)
        data_variant_str = data_variant.name.lower()
        if isinstance(data, list):
            for i, d in enumerate(data):
                self._write_stats(
                    data_variant, d, f'{data_variant_str}-{data_variant.index_to_str(i)}')
            return

        self._write_stats(data_variant, data, data_variant_str)

    def _write_stats(self, data_variant: DataVariant, df: pd.DataFrame, filename: str) -> None:
        data = df.to_numpy()
        mins = np.min(data, axis=0)
        maxs = np.max(data, axis=0)
        means = np.mean(data, axis=0)
        stds = np.std(data, axis=0)

        stats = np.round(np.vstack((mins, maxs, means, stds)), decimals=2)
        columns = list(map(lambda c: c.capitalize(), df.columns))

        df = pd.DataFrame(stats, columns=columns)
        df.insert(0, "Name", pd.Series(["Min", "Max", "Mean", "Std"]))

        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        dir = f'{self.kitti_locations.stats_dir}/{data_variant.name.lower()}'
        os.makedirs(dir, exist_ok=True)
        fpath = f'{dir}/{filename}-{now}'

        df.to_csv(f'{fpath}.csv', index=False)
        # Requires latex installation with booktabs
        # TODO decimal separator
        df.to_latex(
            f'{fpath}.tex',
            float_format="%.2f",
            label=f"table:{filename}",
            position="htb!",
            column_format=len(columns) * "c",
            index=False,
        )

        logging.info(f'Stats written to file:///{fpath}.csv')
