import logging
import os
from typing import Annotated, Optional, Union

import numpy as np
import vtk

import slicer
from MRMLCorePython import vtkMRMLModelNode, vtkMRMLSegmentationNode, vtkMRMLLinearTransformNode, \
    vtkMRMLLabelMapVolumeNode
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode
from vtkSegmentationCorePython import vtkSegmentation
from vtkmodules.vtkCommonDataModel import vtkImageData

from DBS_STN.Lib.image_utils import SlicerImage


#
# LabelConversion
#


class LabelConversion(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("LabelConversion")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "DBS")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#LabelConversion">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # LabelConversion1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="LabelConversion",
        sampleName="LabelConversion1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "LabelConversion1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="LabelConversion1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="LabelConversion1",
    )

    # LabelConversion2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="LabelConversion",
        sampleName="LabelConversion2",
        thumbnailFileName=os.path.join(iconsPath, "LabelConversion2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="LabelConversion2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="LabelConversion2",
    )


#
# LabelConversionParameterNode
#


@parameterNodeWrapper
class LabelConversionParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """

    inputVolume: Union[vtkMRMLModelNode,vtkMRMLLabelMapVolumeNode]
    toMNI: vtkMRMLLinearTransformNode
    imageThreshold: Annotated[float, WithinRange(-100, 500)] = 100
    invertThreshold: bool = False
    thresholdedVolume: vtkMRMLScalarVolumeNode
    invertedVolume: vtkMRMLScalarVolumeNode


#
# LabelConversionWidget
#


class LabelConversionWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/LabelConversion.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = LabelConversionLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.applyConvert.clicked.connect(self.onApplyConvert)
        # Make sure parameter node is initialized (needed for module reload)
        #self.initializeParameterNode()


    def onApplyConvert(self):
        self.logic.convert_label(self.ui.textinputSelector.currentNode(),self.ui.toMNIInputSelector.currentNode(),
                                 self.ui.outputSelector.currentNode())
        pass

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # # Select default input nodes if nothing is selected yet to save a few clicks for the user
        # if not self._parameterNode.inputVolume:
        #     firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        #     if firstVolumeNode:
        #         self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[LabelConversionParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.thresholdedVolume:
            self.ui.applyButton.toolTip = _("Compute output volume")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select input and output volume nodes")
            self.ui.applyButton.enabled = False

    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Compute output
            self.logic.process(self.ui.inputSelector.currentNode(), self.ui.outputSelector.currentNode(),
                               self.ui.imageThresholdSliderWidget.value, self.ui.invertOutputCheckBox.checked)

            # Compute inverted output (if needed)
            if self.ui.invertedOutputSelector.currentNode():
                # If additional output volume is selected then result with inverted threshold is written there
                self.logic.process(self.ui.inputSelector.currentNode(), self.ui.invertedOutputSelector.currentNode(),
                                   self.ui.imageThresholdSliderWidget.value, not self.ui.invertOutputCheckBox.checked, showResult=False)


#
# LabelConversionLogic
#

def getCenterOfMass( volumeNode: vtkMRMLLabelMapVolumeNode):
    """
    Get the center of mass of a binary image
    """
    a = [0,0,0,0,0,0]
    volumeNode.GetRASBounds(a)
    print(a)
    x = np.array([a[0],a[2],a[4]])
    y = np.array([a[1],a[3],a[5]])

    return (x+y)/2




    # # Define some necessary variables for later.
    # centerOfMass = [0,0,0]
    # numberOfStructureVoxels = 0
    # sumX = sumY = sumZ = 0
    #
    # # Gets the image data for the current node.
    # volume = volumeNode
    #
    # # Uses the extent of the image to get the range for the loops,
    # # Then if the value of the given voxel is > zero we add the
    # # value of the voxel coordinate to the running sums, and the
    # # count of voxels is incremented.
    # # We go by 2's to increase the speed - it won't have much (if
    # # any) effect on the result.
    # for z in range(volume.GetExtent()[4], volume.GetExtent()[5] + 1, 2):
    #   for y in range(volume.GetExtent()[2], volume.GetExtent()[3] + 1, 2):
    #     for x in range(volume.GetExtent()[0], volume.GetExtent()[1] + 1, 2):
    #       voxelValue = volume.GetScalarComponentAsDouble(x, y, z, 0)
    #       if voxelValue > 0:
    #         numberOfStructureVoxels = numberOfStructureVoxels + 1
    #         sumX = sumX + x
    #         sumY = sumY + y
    #         sumZ = sumZ + z
    #
    # # When the loop terminates, if we had any voxels, we calculate
    # # the Center of Mass by dividing the sums by the number of voxels
    # # in total.
    # if numberOfStructureVoxels > 0:
    #   centerOfMass[0] = sumX / numberOfStructureVoxels
    #   centerOfMass[1] = sumY / numberOfStructureVoxels
    #   centerOfMass[2] = sumZ / numberOfStructureVoxels
    #
    # print(volume.ComputeCellId([centerOfMass[0],centerOfMass[1],centerOfMass[2]]))
    # # Return the point that contains the center of mass location.
    # return centerOfMass
# def center_of_mass(image_data):
#     # Convert VTK image data to NumPy array
#     image_array = vtk.util.numpy_support.vtk_to_numpy(image_data.GetPointData().GetScalars())
#     image_array = image_array.reshape(image_data.GetDimensions())
#
#     # Get dimensions of the image
#     dims = image_data.GetDimensions()
#
#     # Initialize variables to calculate center of mass
#     total_mass = 0
#     center = np.array([0.0, 0.0, 0.0])
#
#     # Iterate through all voxels to calculate center of mass
#     for z in range(dims[2]):
#         for y in range(dims[1]):
#             for x in range(dims[0]):
#                 # Check if the voxel belongs to the label (assuming label value is 1)
#                 if image_array[x,y,z] == 1:
#                     # Update center of mass
#                     center += np.array([x, y, z])
#                     total_mass += 1
#
#     # Divide by total mass to get the center of mass
#     if total_mass > 0:
#         center /= total_mass
#     else:
#         raise ValueError("No voxels corresponding to the label found in the image.")
#     # convert to from index to coordinate
#     print(center)
#
#     return center


def get_np_from_node(node :vtkMRMLLinearTransformNode):
    mt1 = vtk.vtkMatrix4x4()
    node.GetMatrixTransformToParent(mt1)
    return np.array([[mt1.GetElement(i,j) for j in range(4)] for i in range(4)])


class LabelConversionLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return LabelConversionParameterNode(super().getParameterNode())

    def _generate_vtk_sphere(self, center, radius=10,number_of_divisions=2):
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(center[0],center[1],center[2])

        sphere.SetRadius(radius)
        sphere.Update()
        or_mesh =  sphere.GetOutput()
        # process triugh subdivision filter
        subdivide = vtk.vtkLinearSubdivisionFilter()
        subdivide.SetInputData(or_mesh)
        subdivide.SetNumberOfSubdivisions(number_of_divisions)
        subdivide.Update()
        return subdivide.GetOutput()


    def _compute_final_point(self, sphere_pt: np.ndarray, center:np.ndarray,
                             label_image: SlicerImage, threshold,ras_to_ijk: vtk.vtkMatrix4x4):

        """
        compute final point in native space
            input:
            sphere_pt: point in native space
            center: center of the label in native space
        """

        # get direction vector from pt to center of sphere
        vect = center - sphere_pt
        # normalize vector
        d = np.linalg.norm(vect)
        vect = vect / d

        # generate points along the vector
        points = np.array([sphere_pt + i * (d/100) * vect for i in range(100)])

        # iterate through points and check if the value of the image is above threshold

        values = label_image.compute_image_at_pts(points,ras_to_ijk)
        #print(values)
        #print(points)

        # get the first point that is above threshold
        for i in range(100):
            if values[i] > threshold:
                return points[i]






    def _shrink_sphere_to_label(self, image_data : vtkMRMLLabelMapVolumeNode,
                                sphere: vtk.vtkPolyData,
                                to_mni: np.ndarray,center: np.array,
                                ras_to_ijk: vtk.vtkMatrix4x4):
        """
        shrink sphere to label
        image_data: vtkImageData in native space
        sphere: vtk.vtkPolyData sphere in mni space
        to_mni: transformation matrix from native to mni space
        center: center of the label in native space
        """


        from_mni = np.linalg.inv(to_mni)
        slic_im = SlicerImage(image_data.GetImageData())
        # iterate through all points of the sphere
        all_points = sphere.GetPoints()
        for i in range(sphere.GetNumberOfPoints()):
            point = np.array(list(all_points.GetPoint(i)) + [1])
            # transform point to native space
            point = np.dot(from_mni,point)[:3]
            # get the value of the image at the point

            final_point = self._compute_final_point(point,center,slic_im,0.4,ras_to_ijk)
            all_points.SetPoint(i,final_point[0],final_point[1],final_point[2])
        sphere.SetPoints(all_points)
        return sphere






    def _compute_mesh_from_label_im(self, label_image: vtkMRMLLabelMapVolumeNode,to_mni: vtkMRMLLinearTransformNode):

        # compute center of a label
        center = getCenterOfMass(label_image)
        print(center)
        to_mni_mat = get_np_from_node(to_mni)

        # transform center to MNI space

        center = np.array([center[0],center[1],center[2],1])
        center_mni = np.dot(to_mni_mat,center)[:3]
        # generate sphere in mni
        sphere = self._generate_vtk_sphere(center_mni)

        # shrink sphere to label
        ras2ijk = vtk.vtkMatrix4x4()
        label_image.GetRASToIJKMatrix(ras2ijk)
        sphere = self._shrink_sphere_to_label(label_image,sphere,to_mni_mat,center[:3],ras2ijk)

        return sphere




    def convert_label(self, segmentation_node : Union[vtkMRMLLabelMapVolumeNode,vtkMRMLModelNode,vtkMRMLSegmentationNode],
                      toMNI: vtkMRMLLinearTransformNode, out_node: vtkMRMLModelNode):
        """
        read label from segmentation node  and shrink sphere  to label in MNI space and write to out_node
        """

        # read label from segmentation_node

        if isinstance(segmentation_node,vtkMRMLLabelMapVolumeNode):


            label_image : vtkImageData = segmentation_node.GetImageData()


            a = SlicerImage(label_image)

            transform_ras_to_ijk = vtk.vtkMatrix4x4()
            segmentation_node.GetIJKToRASMatrix(transform_ras_to_ijk)
            transform_ras_to_ijk.Invert()

            resulting_mesh = self._compute_mesh_from_label_im(segmentation_node,toMNI)
            # transform label to MNI space
        else:
            return
        out_node.SetAndObservePolyData(resulting_mesh)
        #label = segmentation_node.GetSegmentation().GetBinaryLabelmapRepresentation("label")






    def process(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                outputVolume: vtkMRMLScalarVolumeNode,
                imageThreshold: float,
                invert: bool = False,
                showResult: bool = True) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """

        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time

        startTime = time.time()
        logging.info("Processing started")

        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        cliParams = {
            "InputVolume": inputVolume.GetID(),
            "OutputVolume": outputVolume.GetID(),
            "ThresholdValue": imageThreshold,
            "ThresholdType": "Above" if invert else "Below",
        }
        cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")


#
# LabelConversionTest
#


class LabelConversionTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_LabelConversion1()

    def test_LabelConversion1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        registerSampleData()
        inputVolume = SampleData.downloadSample("LabelConversion1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = LabelConversionLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")
