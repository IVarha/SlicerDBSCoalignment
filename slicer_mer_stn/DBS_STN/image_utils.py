from dnn_segmentation.image_loader import SubcorticalMask
from vtkmodules import vtkCommonDataModel
import vtk
import numpy as np


class SlicerImage:

    def __init__(self, imageData):
        """

        """
        self.imageData = imageData
        self.interpolator = vtk.vtkImageBSplineCoefficients()
        self.interpolator.SetSplineDegree(3)
        self.interpolator.SetInputData(imageData)
        self.interpolator.Update()

    def compute_value_at_coordinate(self, x, y, z):
        """
        input is a coordinates at t1 coordinates
        """
        return self.interpolator.Evaluate(x, y, z)

    def compute_image_at_pts(self, mask: SubcorticalMask,transform_ras_to_ijk:vtk.vtkMatrix4x4):
        """
        compute image at mask
        return (mirrored and original image)
        """
        coords_origin = np.array(mask.get_coords_list())
        coords_mirrored = np.array([-1, 1, 1]) * np.array(mask.get_coords_list())

        for i in range(coords_origin.shape[0]):
            coords_origin[i,:] =np.array(transform_ras_to_ijk.MultiplyPoint([coords_origin[i,0],
                                                coords_origin[i,1],
                                               coords_origin[i,2],1]))[:3]

            coords_mirrored[i,:] = np.array(transform_ras_to_ijk.MultiplyPoint([coords_mirrored[i,0],
                                                coords_mirrored[i,1],
                                               coords_mirrored[i,2],1]))[:3]

        vect_mirror = []
        vect_orig = []
        for i in range(coords_mirrored.shape[0]):
            vect_o = self.compute_value_at_coordinate(coords_origin[i, 0], coords_origin[i, 1], coords_origin[i, 2])

            vect_m = self.compute_value_at_coordinate(coords_mirrored[i, 0], coords_mirrored[i, 1],
                                                      coords_mirrored[i, 2])

            vect_mirror.append(vect_m)
            vect_orig.append(vect_o)
        shape = (mask.n_x, mask.n_y, mask.n_z)
        return np.reshape(vect_mirror, shape), np.reshape(vect_orig, shape)

    def _compute_label_image_mask(self, coords_list, mirrored, shape):
        """
        computes image within coords in MNI. mirrored or non mirrored
        """
        _im = vt_image.Image(self.image_orig)
        _im.setup_bspline(3)

        if mirrored:
            transf1 = np.dot(self._to_w_label, np.linalg.inv(self.vox_2_mni_mirrored))  # MNI - LABELVOX -> WORLD
            transf1 = np.dot(np.linalg.inv(_im.get_vox_to_world()), transf1)  # -> W2image
            coords_list = utls.apply_transf_2_pts(pts=coords_list, transf=transf1)
        else:

            transf1 = np.dot(self._to_w_label, np.linalg.inv(self.vox_2_mni))  # MNI - LABELVOX -> WORLD
            transf1 = np.dot(np.linalg.inv(_im.get_vox_to_world()), transf1)  # -> W2image
            coords_list = utls.apply_transf_2_pts(pts=coords_list, transf=transf1)  # wrong matrix!!!!

        a = np.array(_im.interpolate_list(coords_list))
        a = a.reshape(shape)
        del _im
        return a
        pass
