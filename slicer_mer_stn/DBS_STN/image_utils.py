from vtkmodules import vtkCommonDataModel
import vtk


class SlicerImage:

    def __init__(self, imageData):
        self.imageData = imageData
        self.interpolator = vtk.vtkImageBSplineCoefficients()
        self.interpolator.SetSplineDegree(3)
        self.interpolator.SetInputData(imageData)
        self.interpolator.Update()

    def compute_value_at_coordinate(self, x, y, z):
        """
        input is a coordinates at t1 coordinates
        """
        return self.interpolator(x, y, z)
