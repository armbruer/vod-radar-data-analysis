import os


class KittiLocations:
    """
    This class contains the information regarding the locations of data for the dataset.
    """
    def __init__(self, 
                 root_dir: str, 
                 output_dir: str = None, 
                 frame_set_path: str = None, 
                 pred_dir: str = None, 
                 data_dir: str = None,
                 stats_dir: str = None,
                 figures_dir: str = None,
                 analysis_dir: str = None,
                 testing: bool = False):
        """
Constructor which based on a few parameters defines the locations of possible data.
        :param root_dir: The root directory of the dataset.
        :param output_dir: Optional parameter of the location where output such as pictures should be generated.
        :param frame_set_path: Optional parameter of the text file of which output should be generated.
        :param pred_dir: Optional parameter of the locations of the prediction labels.
        """

        # Input parameters
        self.root_dir: str = root_dir
        self.output_dir: str = output_dir
        self.frame_set_path: str = frame_set_path
        self.pred_dir: str = pred_dir
        self.data_dir: str = data_dir
        self.stats_dir: str = stats_dir
        self.figures_dir: str = figures_dir
        self.analysis_dir: str = analysis_dir
        self.testing = testing
        
        if testing:
            self.output_dir = f'{self.output_dir}_test'
        
        if self.output_dir is not None:
            if self.stats_dir is None:
                self.stats_dir = f'{self.output_dir}/stats'
            if self.figures_dir is None:
                self.figures_dir = f'{self.output_dir}/figures'
            if self.data_dir is None:
                self.data_dir = f'{self.output_dir}/data'
            if self.analysis_dir is None:
                self.analysis_dir = f'{self.output_dir}/analysis'

        # Automatically defined variables. The location of sub-folders can be customized here.
        # Current definitions are based on the recommended locations.
        self.camera_dir = os.path.join(self.root_dir, 'lidar', 'training', 'image_2')

        self.lidar_dir = os.path.join(self.root_dir, 'lidar', 'training', 'velodyne')
        self.lidar_calib_dir = os.path.join(self.root_dir, 'lidar', 'training', 'calib')

        self.radar_dir = os.path.join(self.root_dir, 'radar', 'training', 'velodyne')
        self.radar_calib_dir = os.path.join(self.root_dir, 'radar', 'training', 'calib')

        self.pose_dir = os.path.join(self.root_dir, 'lidar', 'training', 'pose')
        self.pose_calib_dir = os.path.join(self.root_dir, 'lidar', 'training', 'calib')

        self.label_dir = os.path.join(self.root_dir, 'lidar', 'training', 'label_2')
