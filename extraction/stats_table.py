from typing import List
from extraction import DataVariant, ParameterRangeExtractor
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import os

from vod.configuration.file_locations import KittiLocations


class StatsTableGenerator:

    def __init__(self, kitti_locations: KittiLocations) -> None:
        self.kitti_locations = kitti_locations

    def write_stats(self, data_variant: DataVariant) -> None:
        ex = ParameterRangeExtractor(kitti_locations=self.kitti_locations)
        data: List[np.ndarray] = ex.get_data(data_variant=data_variant)
        data_variant_str = data_variant.name.lower()
        if data_variant == DataVariant.SEMANTIC_OBJECT_DATA_BY_CLASS or data_variant == DataVariant.STATIC_DYNAMIC_RAD:
            for i, d in enumerate(data):
                self._write_stats(
                    data_variant, d, f'{data_variant_str}-{data_variant.index_to_str(i)}')
            return

        self._write_stats(data_variant, *data, data_variant_str)

    def _write_stats(self, data_variant: DataVariant, data: np.ndarray, filename: str) -> None:
        if data.ndim < 2:
            raise ValueError(
                'Dimension of retrieved data must be at least two')

        mins = np.min(data, axis=0)
        maxs = np.max(data, axis=0)
        means = np.mean(data, axis=0)
        stds = np.std(data, axis=0)

        stats = np.vstack((mins, maxs, means, stds))
        columns = data_variant.column_names(with_unit=True)
        columns = list(map(lambda c: c.capitalize(), columns))

        df = pd.DataFrame(stats, columns=columns)
        df.insert(0, "Name", pd.Series(["Min", "Max", "Mean", "Var", "Std"]))

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
