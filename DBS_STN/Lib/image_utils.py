from vtkmodules import vtkCommonDataModel
import vtk
import numpy as np


from dbs_image_utils.mask import SubcorticalMask
from vtkmodules.vtkCommonDataModel import vtkImageData


class SlicerImage:

    def __init__(self, imageData: vtkImageData):
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


    def compute_image_at_mask(self, mask: SubcorticalMask, transform_ras_to_ijk: vtk.vtkMatrix4x4):
        """
        compute image at mask
        return (mirrored and original image)
        """
        coords_origin = np.array(mask.get_coords_list())
        coords_mirrored = np.array([-1, 1, 1]) * np.array(mask.get_coords_list())

        for i in range(coords_origin.shape[0]):
            coords_origin[i, :] = np.array(transform_ras_to_ijk.MultiplyPoint([coords_origin[i, 0],
                                                                               coords_origin[i, 1],
                                                                               coords_origin[i, 2], 1]))[:3]

            coords_mirrored[i, :] = np.array(transform_ras_to_ijk.MultiplyPoint([coords_mirrored[i, 0],
                                                                                 coords_mirrored[i, 1],
                                                                                 coords_mirrored[i, 2], 1]))[:3]

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

    def compute_image_at_pts(self, points, transform_ras_to_ijk: vtk.vtkMatrix4x4):
        """
        compute image at points defined in image coordinates
        return (mirrored and original image)
        """
        coords_origin = np.array(points)

        for i in range(coords_origin.shape[0]):
            coords_origin[i, :] = np.array(transform_ras_to_ijk.MultiplyPoint([coords_origin[i, 0],
                                                                               coords_origin[i, 1],
                                                                               coords_origin[i, 2], 1]))[:3]

        vect_orig = []
        for i in range(coords_origin.shape[0]):
            vect_o = self.compute_value_at_coordinate(coords_origin[i, 0], coords_origin[i, 1], coords_origin[i, 2])

            vect_orig.append(vect_o)
        return vect_orig

