import logging
import os
from typing import Annotated, Optional
import slicer
import vtk

import slicer
from MRMLCorePython import vtkMRMLModelNode, vtkMRMLTransformNode, vtkMRMLLabelMapVolumeNode
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)
import numpy as np
from slicer import vtkMRMLScalarVolumeNode

import os


import numpy as np


def load_segmentation(nifti_path):
    # Load NIfTI segmentation file
    segmentation_node = slicer.util.loadSegmentation(nifti_path)
    return segmentation_node


def adjust_segment_colors(segmentation_node, color_map):
    # Get segmentation display node
    segmentation_display_node = segmentation_node.GetDisplayNode()

    # Update segment colors
    for segment_name, color in color_map.items():
        segment_id = segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(segment_name)
        segmentation_display_node.SetSegmentColor(segment_id, color)



def convert_model_to_segmentation(model_node) -> vtkMRMLLabelMapVolumeNode:
    # Create a new segmentation node
    segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    segmentation_node.CreateDefaultDisplayNodes()  # Create display nodes

    # Create a segment within the segmentation node
    #segment_id = segmentation_node.GetSegmentation().AddEmptySegment()
    #segment = segmentation_node.GetSegmentation().GetSegment(segment_id)

    # Render the model into a labelmap volume
    labelmap_volume_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    print(model_node)
    print(segmentation_node)
    slicer.modules.segmentations.logic().ImportModelToSegmentationNode(model_node, segmentation_node)
    slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(segmentation_node,
                                                                      labelmap_volume_node)
    # # Threshold the labelmap volume to obtain a binary labelmap
    # threshold = slicer.vtkImageThreshold()
    # threshold.SetInputData(labelmap_volume_node.GetImageData())
    # threshold.ThresholdByLower(1)  # Set the threshold value to 1
    # threshold.SetInValue(1)
    # threshold.SetOutValue(0)
    # threshold.Update()
    #
    # # Update the labelmap volume with the thresholded data
    # labelmap_volume_node.SetAndObserveImageData(threshold.GetOutput())


    return labelmap_volume_node

def resourcePath( relativePath):
    """
        Get the absolute path to the module resource
    """
    dirn = os.path.dirname(__file__)
    print("pt1", dirn)
    res = os.path.join(dirn, "Resources", relativePath)
    print("pt2", res)
    return res


def add_empty_voxels_nifti(nifti_image, num_empty_voxels):
    # Get the data array from the NIfTI image
    try:
        import nibabel as nib
    except ImportError:
        slicer.util.pip_install("nibabel")
        import nibabel as nib

    image_data = nifti_image.get_fdata()

    # Get the dimensions of the original image
    original_shape = image_data.shape

    # Ensure num_empty_voxels is an integer
    num_empty_voxels = int(num_empty_voxels)

    # Calculate the new shape with additional empty voxels
    new_shape = tuple(np.array(original_shape) + 2 * num_empty_voxels)

    # Create a larger array with empty voxels
    larger_image_data = np.zeros(new_shape)

    # Calculate the indices to copy the original image into the larger array
    start_indices = tuple(num_empty_voxels for _ in range(len(original_shape)))
    end_indices = tuple(num_empty_voxels + s for s in original_shape)

    # Copy the original image into the larger array
    larger_image_data[start_indices[0]:end_indices[0], start_indices[1]:end_indices[1],
    start_indices[2]:end_indices[2]] = image_data

    # Update the origin to account for the shift
    old_origin = nifti_image.affine[:3, 3]
    new_origin = old_origin - np.array(num_empty_voxels) * nifti_image.header.get_zooms()[:3]
    new_affine = np.copy(nifti_image.affine)
    new_affine[:3, 3] = new_origin

    # Create a new NIfTI image with the larger data array and updated origin
    larger_nifti_image = nib.Nifti1Image(larger_image_data, new_affine)

    return larger_nifti_image

#
# AtlasMapping
#


class AtlasMapping(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("AtlasMapping")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#AtlasMapping">module documentation</a>.
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

    # AtlasMapping1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="AtlasMapping",
        sampleName="AtlasMapping1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "AtlasMapping1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="AtlasMapping1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="AtlasMapping1",
    )

    # AtlasMapping2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="AtlasMapping",
        sampleName="AtlasMapping2",
        thumbnailFileName=os.path.join(iconsPath, "AtlasMapping2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="AtlasMapping2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="AtlasMapping2",
    )


def write_points_to_file(poly_data, filename):
    points = poly_data.GetPoints()

    with open(filename, 'w') as f:
        f.write("point\n")
        f.write(str(points.GetNumberOfPoints()) + "\n")
        for i in range(points.GetNumberOfPoints()):
            point = points.GetPoint(i)
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

#
# AtlasMappingParameterNode
#


@parameterNodeWrapper
class AtlasMappingParameterNode:
    """
    The parameters needed by module.

    inputMesh - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """

    inputMesh: vtkMRMLModelNode
    invertedVolume: vtkMRMLTransformNode


#
# AtlasMappingWidget
#


class AtlasMappingWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
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
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/AtlasMapping.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = AtlasMappingLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

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

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputMesh:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLModelNode")
            if firstVolumeNode:
                self._parameterNode.inputMesh = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[AtlasMappingParameterNode]) -> None:
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
        if self._parameterNode and self._parameterNode.inputMesh:
            self.ui.applyButton.toolTip = _("Compute output volume")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select input and output volume nodes")
            self.ui.applyButton.enabled = False

    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Compute output


            segment_colors = {
                "STN_motor": [255, 105, 240],  # Red
                "STN_limbic": [255, 24, 9],  # Green
                "STN_associative": [53, 255, 255]  # Blue
            }
            segme_paths = [resourcePath("eparams/STN_motor.nii.gz"),
                           resourcePath("eparams/STN_limbic.nii.gz"),resourcePath("eparams/STN_associative.nii.gz")]


            def_name = "Segment_1"
            segmentation_nodes = [load_segmentation(nifti_path) for nifti_path in segme_paths]
            seg_1 = segmentation_nodes[0]
            seg_1.GetSegmentation().GetSegment(def_name).SetColor(segment_colors["STN_motor"])
            seg_1.GetSegmentation().GetSegment(def_name).SetName("STN_motor")
            # add segments to the segmentation node
            seg_1.GetSegmentation().AddSegment(segmentation_nodes[1].GetSegmentation().GetSegment(def_name))
            seg_1.GetSegmentation().GetSegment(seg_1.GetSegmentation().GetSegmentIdBySegmentName(def_name)).SetColor(segment_colors["STN_limbic"])
            seg_1.GetSegmentation().GetSegment(seg_1.GetSegmentation().GetSegmentIdBySegmentName(def_name)).SetName("STN_limbic")
            seg_1.GetSegmentation().AddSegment(segmentation_nodes[2].GetSegmentation().GetSegment(def_name))
            seg_1.GetSegmentation().GetSegment(seg_1.GetSegmentation().GetSegmentIdBySegmentName(def_name)).SetColor(
                segment_colors["STN_associative"])
            seg_1.GetSegmentation().GetSegment(seg_1.GetSegmentation().GetSegmentIdBySegmentName(def_name)).SetName("STN_associative")
            seg_1.SetName("STN_Accolla")

            # remove the other segmentations
            for seg in segmentation_nodes[1:]:
                slicer.mrmlScene.RemoveNode(seg)
            meshN = self.ui.inputSelector.currentNode()
            transf_id = meshN.GetTransformNodeID()
            if transf_id: #reset the transform node as it wouldnt work.
                meshN.SetAndObserveTransformNodeID(None)
            self.logic.process(meshN, self.ui.invertedOutputSelector.currentNode())

            if transf_id:
                meshN.SetAndObserveTransformNodeID(transf_id)
                # set and observe transform node to invertedOutputSelector
                self.ui.invertedOutputSelector.currentNode().SetAndObserveTransformNodeID(meshN.GetTransformNodeID())

            # set and observe transform node to segment
            seg_1.SetAndObserveTransformNodeID(self.ui.invertedOutputSelector.currentNode().GetID())
            # adjust_segment_colors(seg_1, segment_colors)

#
# AtlasMappingLogic
#


class AtlasMappingLogic(ScriptedLoadableModuleLogic):
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
        return AtlasMappingParameterNode(super().getParameterNode())

    def process(self,
                inputMesh: vtkMRMLModelNode,
                outputVolume: vtkMRMLTransformNode) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputMesh: volume to be thresholded
        :param outputVolume: thresholding result
        """

        if not inputMesh or not outputVolume:
            raise ValueError("Input or output volume is invalid")
        try:
            import nibabel as nib
        except ImportError:
            slicer.util.pip_install("nibabel")
            import nibabel as nib

        import time
        startTime = time.time()
        # generate temp folder
        import tempfile
        dir = tempfile.TemporaryDirectory()
        print("temp dir", dir.name)
        try:
            # convert model to segmentation
            segmentation = convert_model_to_segmentation(inputMesh)

            # save segmentation to nifti using slicer
            slicer.util.saveNode(segmentation, os.path.join(dir.name, "mesh_label.nii.gz"))
            # add extra voxels to nifti

            nifti_image = nib.load(os.path.join(dir.name, "mesh_label.nii.gz"))
            nifti_image = add_empty_voxels_nifti(nifti_image, 40)
            nib.save(nifti_image, os.path.join(dir.name, "mesh_label.nii.gz"))

            # save mesh control points to file
            write_points_to_file(inputMesh.GetPolyData(), os.path.join(dir.name, "mesh_points.txt"))

            # coregistration run
            import Elastix
            elastix = Elastix.ElastixLogic()

            elastixParams = [
                              "-f", os.path.join(dir.name, "mesh_label.nii.gz"),
                              "-fp", os.path.join(dir.name, "mesh_points.txt"),
                              "-m", resourcePath("eparams/STNlabel.nii.gz"),
                              "-mp", resourcePath("eparams/atlas_pts.txt"),
                              "-p", resourcePath("eparams/affine.txt"),
                              "-p", resourcePath("eparams/bspline.txt"),
                              "-out", dir.name]
            print( " ".join(elastixParams))
            ep = elastix.startElastix(elastixParams)
            elastix.logProcessOutput(ep)
            # edit transformix parameters
            transform_params = [
                "-in", resourcePath(f"eparams{os.path.sep}STNlabel.nii.gz"),
                "-tp", os.path.join(dir.name, "TransformParameters.1.txt"),
                "-def","all",
                "-out", dir.name
            ]
            tp = elastix.startTransformix(transform_params)
            # load transformation
            elastix.logProcessOutput(tp)
            outputTransformPath = os.path.join(dir.name, "deformationField.nii.gz")
            elastix.loadTransformFromFile(outputTransformPath, outputVolume)
        except Exception as e:
            raise ValueError(e)
        finally:
            pass
            #dir.cleanup()


        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")


#
# AtlasMappingTest
#


class AtlasMappingTest(ScriptedLoadableModuleTest):
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
        self.test_AtlasMapping1()

    def test_AtlasMapping1(self):
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
        inputMesh = SampleData.downloadSample("AtlasMapping1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputMesh.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = AtlasMappingLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputMesh, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputMesh, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")
