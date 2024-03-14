from pathlib import Path
from typing import List

import numpy as np
from sklearn.base import TransformerMixin


class CenterMeshPoints(TransformerMixin):

    def fit(self, X):
        return self

    def transform(self, X):
        arr = []
        for label in X:
            ctr = np.array(label.center_of_mesh())
            pts = label.get_unpacked_coords()

            pts = np.array(pts).reshape((int(len(pts) / 3), 3))
            pts = pts - ctr
            pts = pts.reshape((-1))

            # print(len(pts))
            # return
            arr.append(pts)
        return np.array(arr)

    def inverse_transform(self, X):
        return X


def get_images_in_folder(folder:str):
    fold = Path(folder)


    return [ x.name for x in fold.iterdir()
             if (x.name.endswith('nii.gz') or x.name.endswith('.nii'))
             ]




