from typing import Optional, List


class FrameLabels:
    """
    This class is responsible for converting the label string list to a list of Python dictionaries.
    """

    def __init__(self,
                 raw_labels: List[str]):
        """
        Constructor which creates the label property, given a list of strings containing the label data.
        :param raw_labels: List of strings containing label data.
        """
        self.raw_labels: List[str] = raw_labels

        self._labels_dict: Optional[List[dict]] = None
        
        self._complete_labels_dict: Optional[List[dict]] = None

    @property
    def labels_dict(self):
        """
        Label dictionary property.
        :return:
        """
        if self._labels_dict is not None:
            # When the data is already loaded.
            return self._labels_dict
        else:
            # Load data if it is not loaded yet.
            self._labels_dict = self.get_labels_dict()
            return self._labels_dict
        
    @property
    def complete_labels_dict(self):
        """
        Label dictionary property containg all labels
        :return:
        """
        if self._complete_labels_dict is not None:
            # When the data is already loaded.
            return self._complete_labels_dict
        else:
            # Load data if it is not loaded yet.
            self._complete_labels_dict = self.get_labels_dict(complete=True)
            return self._complete_labels_dict

    def get_labels_dict(self, complete=False) -> List[dict]:
        """
        This method returns a list of dictionaries containing the label data.
        :return: List of dictionaries containing label data.
        """


        labels = []  # List to be filled

        for act_line in self.raw_labels:  # Go line by line to split the keys
            act_line = act_line.split()
            label, tracking_id, occl, alpha, left, top, right, bottom, h, w, l, x, y, z, rot, score = act_line
            alpha, left, top, right, bottom = map(float, [alpha, left, top, right, bottom])
            h, w, l, x, y, z, rot, score = map(float, [h, w, l, x, y, z, rot, score])
            tracking_id, occl = map(int, [tracking_id, occl])
            

            labels_dict = {'label_class': label,
                           'h': h,
                           'w': w,
                           'l': l,
                           'x': x,
                           'y': y,
                           'z': z,
                           'rotation': rot,
                           'score': score}
            
            if complete:
                labels_dict.update({
                           'tracking_id': tracking_id,
                           'occl': occl,
                           'alpha': alpha,
                           'l': left,
                           't': top,
                           'r': right,
                           'b': bottom})
            
            labels.append(labels_dict)
            
        return labels


