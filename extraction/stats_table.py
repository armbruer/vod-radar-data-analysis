from itertools import groupby
import logging
import os
import numpy as np
import pandas as pd

from extraction.file_manager import DataManager, DataView
from extraction.helpers import DataVariant, DataViewType, from_class_label_to_summarized_name
from vod.common.file_handling import get_frame_list_from_folder
from vod.frame.data_loader import FrameDataLoader
from vod.frame.labels import FrameLabels


class StatsTableGenerator:

    def __init__(self, data_manager: DataManager) -> None:
        self.data_manager = data_manager
        self.kitti_locations = data_manager.kitti_locations

    def write_stats(self, data_variant: DataVariant) -> None:
        data_view: DataView = self.data_manager.get_view(data_variant=data_variant, data_view_type=DataViewType.STATS)
        data = data_view.df
        data_variant_str = data_variant.shortname()
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

        dir = f'{self.kitti_locations.stats_dir}/{data_variant.shortname()}'
        os.makedirs(dir, exist_ok=True)
        fpath = f'{dir}/{filename}'

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

        logging.info(f'Stats written to "file:///{fpath}.csv"')
        
    def write_class_counter(self):
        data_view: DataView = self.data_manager.get_view(data_variant=DataVariant.SEMANTIC_DATA, 
                                                         data_view_type=DataViewType.NONE)
        df: pd.DataFrame = data_view.df
        
        # only those that have at least one detection will be counted
        # see extraction algorithm in extract.py
        df_counts_only_detections = df["Data Class"].value_counts()
        
        # all objects, even those with zero detections
        df_counts_all: pd.DataFrame = self._get_all_semantic_data()
        
        dir = f'{self.kitti_locations.stats_dir}'
        os.makedirs(dir, exist_ok=True)
        
        to_latex = lambda df, dir, filename: df.to_latex(
            f'{dir}/{filename}.tex',
            float_format="%.2f",
            label=f"table:{filename}",
            position="htb!",
            column_format=2 * "c",
            index=False,
        )
        
        to_latex(df_counts_all, dir, "counts_all")
        to_latex(df_counts_only_detections, dir, "counts_detections_only")
        
    
    def _get_all_semantic_data(self) -> pd.DataFrame:
        # only those frame_numbers which have annotations
        frame_numbers = get_frame_list_from_folder(self.kitti_locations.label_dir)

        all_labels = []
        for frame_number in frame_numbers:
            loader = FrameDataLoader(kitti_locations=self.kitti_locations, frame_number=frame_number)
            
            labels = loader.get_labels()
            if labels is None:
                continue
            
            labels = FrameLabels(labels)
            all_labels += labels.labels_dict
        
        
        no_dont_care = list(filter(lambda x: x['label_class'] != 'DontCare', all_labels))
        for x in no_dont_care:
            x['label_class'] = from_class_label_to_summarized_name(x['label_class'])
        all_labels = sorted(no_dont_care, key=lambda x: x['label_class']) 
        class_count = {k: len(list(v)) for k, v in 
                       groupby(all_labels, key=lambda x: x['label_class'])}
        class_count = dict(sorted(class_count.items(), key=lambda item: item[1], reverse=True))
        
        return pd.DataFrame(data=class_count, index=[0])
        
        
def generate_stats(dm : DataManager):
    stats_generator = StatsTableGenerator(dm)
    
    for dv in DataVariant.all_variants():
        stats_generator.write_stats(dv)
