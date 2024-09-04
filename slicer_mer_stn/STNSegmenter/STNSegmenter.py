import logging
import os
import shlex
from typing import Annotated, Optional
import logging
import slicer

import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated, Optional, Tuple

from MRMLCorePython import vtkMRMLVolumeArchetypeStorageNode, vtkMRMLTransformNode, vtkMRMLModelNode, \
    vtkMRMLModelDisplayNode
try:
    from segm_lib import slicer_preprocessing
    from segm_lib.image_utils import SlicerImage
except ImportError:
    slicer.util.pip_install(r'nibabel')
    slicer.util.pip_install('intensity-normalization')
    if sys.platform == 'win32':
        slicer.util.pip_install('antspyx')
        slicer.util.pip_install('antspynet')

    from segm_lib import slicer_preprocessing
    from segm_lib.image_utils import SlicerImage

try:
    from dbs_image_utils.mask import SubcorticalMask
except ImportError:

    #slicer.util.pip_install('dbs-image-utils')
    from dbs_image_utils.mask import SubcorticalMask
from dbs_image_utils.nets import CenterDetector, CenterAndPCANet, TransformerShiftPredictor, TransformerClassifier

try:
    import fsl.data.image as fim
except ImportError:
    slicer.util.pip_install('fslpy')
    import fsl.data.image as fim
import fsl.transform.flirt as fl

try:
    import mer_lib.artefact_detection as ad
except ImportError:
    #slicer.util.pip_install(r'C:\\Users\\h492884\\PycharmProjects\\MER_lib') todo add installation
    import mer_lib.artefact_detection as ad

if sys.platform == 'win32':
    try:
        import ants
        import antspynet
    except ImportError:
        slicer.util.pip_install('tensorflow==2.14.0')
        slicer.util.pip_install('tensorflow-estimator==2.11.0')

        slicer.util.pip_install('tensorflow-probability==0.22.1')
        slicer.util.pip_install('keras==2.10.0')
        slicer.util.pip_install('antspyx')

import mer_lib.feature_extraction as fe
import mer_lib.processor as proc
import numpy as np
import qt
import vtk

try:
    import torch
except ImportError:
    slicer.util.pip_install('torch')
    import torch
try:
    import intensity_normalization as inorm
except ImportError:
    slicer.util.pip_install('intensity-normalization')
    import intensity_normalization as inorm

import vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode


#
# STNSegmenter
#
def read_mesh(file_name):
    mesh = vtk.vtkOBJReader()
    mesh.SetFileName(file_name)
    mesh.Update()
    mesh = mesh.GetOutput()
    return mesh


def _read_pickle(filename):
    f = open(filename, 'rb')
    res = pickle.load(f)
    f.close()
    return res


def loadNiiImage(file_path):
    # Load an image and display it in Slicer
    image_node = slicer.util.loadVolume(file_path)
    slicer.util.setSliceViewerLayers(background=image_node)
    return image_node


def _compute_min_max_scaler(pt_min, pt_max):
    try:
        from sklearn.preprocessing import MinMaxScaler
    except ImportError:
        slicer.util.pip_install('scikit-learn')
        from sklearn.preprocessing import MinMaxScaler
    a = MinMaxScaler()
    a.fit([pt_min, pt_max])
    return a


class STNSegmenter(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("STNSegmenter")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "DBS")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = [
            "John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#STNSegmenter">module documentation</a>.
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

    # STNSegmenter1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="STNSegmenter",
        sampleName="STNSegmenter1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "STNSegmenter1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="STNSegmenter1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="STNSegmenter1",
    )

    # STNSegmenter2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="STNSegmenter",
        sampleName="STNSegmenter2",
        thumbnailFileName=os.path.join(iconsPath, "STNSegmenter2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="STNSegmenter2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="STNSegmenter2",
    )


#
# STNSegmenterParameterNode
#


@parameterNodeWrapper
class STNSegmenterParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """

    inputVolume: vtkMRMLScalarVolumeNode
    imageThreshold: Annotated[float, WithinRange(-100, 500)] = 100
    invertThreshold: bool = False
    thresholdedVolume: vtkMRMLScalarVolumeNode
    invertedVolume: vtkMRMLScalarVolumeNode


#
# STNSegmenterWidget
#


def check_storage_node(node: vtkMRMLScalarVolumeNode,
                       temp_workdir: tempfile.TemporaryDirectory) -> vtkMRMLVolumeArchetypeStorageNode:
    storageNode = node.GetStorageNode()
    if storageNode is None:  # save node to temp folder and return storage node for it
        slicer.util.saveNode(node, str(Path(temp_workdir.name) / f"{node.GetName()}.nii.gz"))
        return node.GetStorageNode()
    elif (storageNode.GetFileName()) and not storageNode.GetFileName().endswith(".nii.gz"):
        slicer.util.saveNode(node, str(Path(temp_workdir.name) / f"{node.GetName()}.nii.gz"))
        return node.GetStorageNode()
    elif not storageNode.GetFileName():
        slicer.util.saveNode(node, str(Path(temp_workdir.name) / f"{node.GetName()}.nii.gz"))
        return node.GetStorageNode()
    return storageNode


class STNSegmenterWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
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
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/STNSegmenter.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = STNSegmenterLogic()
        self._create_temp_folder()
        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.preprocessingButton.clicked.connect(self.onApplyPreprocessing)
        self.ui.betButton.clicked.connect(self.brain_extraction)
        self.ui.wmSegmentationButton.clicked.connect(self.onApplyWMSeg)
        self.ui.wmIntensityNormButton.clicked.connect(self.onApplyIntensity)
        self.ui.twoStepCoregistrationButton.clicked.connect(self.onTwoStepCoregistration)
        self.ui.segmentationButton.clicked.connect(self.onSegmentationButtonClicked)

        self.ui.t2inputSelector.currentNodeChanged.connect(lambda x: self.onVolumeSelect(x, "t2_node"))
        self.ui.inputSelector.currentNodeChanged.connect(lambda x: self.onVolumeSelect(x, "t1_node"))
        self.t1_node = None
        self.t2_node = None

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def onVolumeSelect(self, x: vtkMRMLScalarVolumeNode, name):
        # get storage node of x
        if x is not None:
            setattr(self, name, x)
        pass

    def _create_temp_folder(self):
        self.temp_workdir = tempfile.TemporaryDirectory()
        print(self.temp_workdir.name)

    def onApplyPreprocessing(self) -> None:
        print("start on appl")

        sn1 = check_storage_node(self.t1_node, self.temp_workdir)
        sn2 = check_storage_node(self.t2_node, self.temp_workdir)
        print(sn1.GetFileName())
        print(sn2.GetFileName())

        self.logic.coregistration_t2_t1(sn1
                                        , t2=sn2,
                                        out_name=str(Path(self.temp_workdir.name) / "coreg_t2.nii.gz"))

        # load t2 coregistered image
        t2_node = loadNiiImage(str(Path(self.temp_workdir.name) / "coreg_t2.nii.gz"))
        self.ui.t2inputSelector.currentNodeChanged(t2_node)
        print(self.t2_node.GetName())
        self.popup_window()
        print("fin on appl")

    def popup_window(self):
        message_box = qt.QMessageBox()

        # Set the message box type (information, warning, etc.)
        message_box.setIcon(qt.QMessageBox.Information)

        # Set the title and message text
        message_box.setWindowTitle("Process Completed")
        message_box.setText("The process has finished successfully.")

        # Add an "OK" button to the message box
        message_box.addButton(qt.QMessageBox.Ok)

        # Show the message box as a modal dialog
        message_box.exec_()

    def onApplyWMSeg(self) -> None:
        print("start on onApplyWMSeg")

        slicer_preprocessing.wm_segmentation(t1=str(Path(self.temp_workdir.name) / "t1.nii.gz"),
                                             out_folder=self.temp_workdir.name)
        self.wm_seg_done = True
        print("fin on appl")

    def onApplyIntensity(self):
        print("test onApplyIntensity")
        # slicer_preprocessing.intensity_normalisation(self.temp_workdir.name)
        if self.wm_seg_done:
            self.logic.intensity_normalisation(self.temp_workdir.name)
            self.intensity_normalisation_done = True
            t2_node = loadNiiImage(str(Path(self.temp_workdir.name) / "t2_normalised.nii.gz"))
            self.ui.t2inputSelector.currentNodeChanged(t2_node)

            pass
        self.popup_window()

    def onTwoStepCoregistration(self):
        print("test onTwoStepCoregustration")

        self.transform_node = self.logic.two_step_coregistration(self.t2_node, self.temp_workdir.name)
        self.transform_node.SetName("to_mni")

    def apply_normalization(self, shape_im):
        shape_im = self.shape_histogram.apply_normalization(shape_im)
        return shape_im

    def brain_extraction(self):

        self.logic.brain_extraction(check_storage_node(self.t1_node, self.temp_workdir), self.temp_workdir.name)

        t1_node = loadNiiImage(str(Path(self.temp_workdir.name) / "t1.nii.gz"))

        self.ui.inputSelector.currentNodeChanged(t1_node)

    def onSegmentationButtonClicked(self):
        mm_offset = 2
        print("Starting segmentation")

        left, right = self.logic.segmentSTNs(self.t2_node)

        ## add mni^-1 transformation to the mesh nodes

        # invert tranform node
        inverted_transform = slicer.vtkMRMLTransformNode()
        inverted_transform.SetName("to_mni_inverted")
        mt1 = vtk.vtkMatrix4x4()

        self.transform_node.GetMatrixTransformFromParent(mt1)
        inverted_transform.SetMatrixTransformToParent(mt1)
        slicer.mrmlScene.AddNode(inverted_transform)
        # inverted_transform.SetMatrixTransformToParent(self.transform_node.GetMatrixTransformToParent())

        left[0].SetAndObserveTransformNodeID(inverted_transform.GetID())
        right[0].SetAndObserveTransformNodeID(inverted_transform.GetID())
        self.t2_node.SetAndObserveTransformNodeID(inverted_transform.GetID())

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
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[STNSegmenterParameterNode]) -> None:
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
                                   self.ui.imageThresholdSliderWidget.value, not self.ui.invertOutputCheckBox.checked,
                                   showResult=False)


#
# STNSegmenterLogic
#

MESH_results = Tuple[vtkMRMLModelNode, np.ndarray]


class STNSegmenterLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def resourcePath(self, relativePath):
        """
        Get the absolute path to the module resource
        """
        # print("pt1", os.path.dirname(__file__))
        return os.path.normpath(os.path.join(os.path.dirname(__file__), "Resources", relativePath))

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)
        try:
            import torchio
        except ImportError:
            slicer.util.pip_install('torchio')
            import torchio

        self.cent_det_hist = _read_pickle(self.resourcePath('nets/cent_detect_hist.pkl'))

        self.det_mask: SubcorticalMask = _read_pickle(self.resourcePath('nets/detect_mask.pkl'))
        self.processing_folder = None
        self.center_detector_scaller = _compute_min_max_scaler(self.det_mask.min_p, self.det_mask.max_p)
        net = CenterDetector().to('cpu')
        cd_state_dict = torch.load(self.resourcePath('nets/cent_pred.pt'), map_location=torch.device('cpu'))
        net.load_state_dict(cd_state_dict)
        self.center_detector = net

        self.shape_pca_res = _read_pickle(self.resourcePath('nets/shape_pcas.pkl'))

        self.shape_label_mask: SubcorticalMask = _read_pickle(self.resourcePath('nets/stn_shape_mask.pkl'))

        # load segmentation model
        self.shape_histogram = _read_pickle(self.resourcePath('nets/shape_hist.pkl'))
        net = CenterAndPCANet(self.shape_pca_res[1])
        cd_state_dict = torch.load(self.resourcePath('nets/shp_pred.pt'), map_location=torch.device('cpu'))
        net.load_state_dict(cd_state_dict)
        self.shape_predictor = net

    def getParameterNode(self):
        return STNSegmenterParameterNode(super().getParameterNode())

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
        cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True,
                                 update_display=showResult)
        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime - startTime:.2f} seconds")

    def brain_extraction(self, t1: vtkMRMLVolumeArchetypeStorageNode, temp_dir_path) -> None:
        image_name = t1.GetFileName()
        mask_filename = str(Path(temp_dir_path) / "t1_mask.nii.gz")

        if sys.platform == 'win32':

            img = ants.image_read(image_name)

            mask = antspynet.brain_extraction(img, "t1") > 0.8
            masked_image = img * mask
            ants.image_write(mask, mask_filename)
            ants.image_write(masked_image, str(Path(temp_dir_path) / "t1.nii.gz"))
        elif sys.platform == 'darwin':
            cmd = "python -c 'import antspynet;import ants;from pathlib import Path;"
            cmd += f"img=ants.image_read(\"{image_name}\");"
            cmd += f"mask=antspynet.brain_extraction(img, \"t1\") > 0.8;"
            cmd += f"masked_image=img*mask;"
            cmd += f"ants.image_write(mask, \"{mask_filename}\");"
            cmd += f"ants.image_write(masked_image, \"{str(Path(temp_dir_path) / 't1.nii.gz')}\")'"
            cmd = shlex.split(cmd)
            subprocess.check_output(cmd)

        print("FINISHED EXTRACTOR")
        # cmd = [sys.executable, self.resourcePath("py/bet.py"), str(image_name), mask_filename, str(Path(temp_dir_path) / "t1.nii.gz")]
        # print(cmd)
        # subprocess.call(cmd, shell=True)

    def coregistration_t2_t1(self, t1: vtkMRMLVolumeArchetypeStorageNode, t2: vtkMRMLVolumeArchetypeStorageNode,
                             out_name: str) -> None:

        out_folder = str(Path(out_name).parent)
        t1_path = t1.GetFileName()
        t2_path = t2.GetFileName()
        print(t1_path)
        print(t2_path)
        # Elastix.ElastixLogic().register()
        slicer_preprocessing.elastix_registration(
            ref_image=t1_path,
            flo_image=t2_path,
            elastix_parameters=self.resourcePath('elastix/rigid_mri.txt'),
            out_folder=out_folder)
        ((Path(out_folder) / "result.0.nii.gz")
         .rename((out_name)))

    def wm_segmentation(self, t1: str, out_folder: str) -> None:
        slicer_preprocessing.wm_segmentation(t1, out_folder)

    def intensity_normalisation(self, out_folder: str) -> None:
        slicer_preprocessing.intensity_normalisation(out_folder)

    def two_step_coregistration(self, node_to_transform, workdir: str) -> vtkMRMLTransformNode:
        mni = self.resourcePath('MNI/MNI152_T1_1mm_brain.nii.gz')
        elastix_affine = self.resourcePath('elastix/affine_mri.txt')
        struct_image = str(Path(workdir) / "t1.nii.gz")
        slicer_preprocessing.elastix_registration(ref_image=mni,
                                                  flo_image=struct_image,
                                                  elastix_parameters=elastix_affine,
                                                  out_folder=workdir)
        tfm_file = Path(workdir) / "TransformParameters.0.tfm"
        transfortm_node = slicer.util.loadTransform(tfm_file)

        node_to_transform.SetAndObserveTransformNodeID(transfortm_node.GetID())
        node_to_transform.HardenTransform()
        return transfortm_node

    def segmentSTNs(self, t2_node) -> Tuple[MESH_results, MESH_results]:
        mm_offset = 2
        print("Starting segmentation")

        t2 = t2_node
        image_processor = SlicerImage(t2.GetImageData())

        transform_ras_to_ijk = vtk.vtkMatrix4x4()
        t2.GetIJKToRASMatrix(transform_ras_to_ijk)

        transform_ras_to_ijk.Invert()
        a = image_processor.compute_image_at_mask(self.det_mask, transform_ras_to_ijk)
        a = list(a)
        a[0] = self.cent_det_hist.apply_normalization(a[0])
        a[1] = self.cent_det_hist.apply_normalization(a[1])
        try:
            res_a0 = np.expand_dims(a[0], axis=0)
            res_a0 = self.center_detector(torch.from_numpy(np.expand_dims(res_a0, axis=0)))
            res_a1 = np.expand_dims(a[1], axis=0)
            res_a1 = self.center_detector(torch.from_numpy(np.expand_dims(res_a1, axis=0)))

            res_a0 = self.center_detector_scaller.inverse_transform(res_a0.detach().numpy())[0]
            res_a1 = self.center_detector_scaller.inverse_transform(res_a1.detach().numpy())[0]
            self.center_orig = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode')
            self.center_orig.AddControlPointWorld(-1 * res_a0[0], res_a0[1], res_a0[2])
            self.center_orig.AddControlPointWorld(res_a1[0], res_a1[1], res_a1[2])
            # compute segmentation
            image_coords_mirr = self.shape_label_mask.get_coords_list() + res_a0
            image_coords_orig = self.shape_label_mask.get_coords_list() + res_a1
            image_coords_mirr = image_coords_mirr * np.array([-1, 1, 1])

            mesh_orig, cent_orig, pts_left = self.segment_side(t2, res_a1, image_coords_orig)
            mesh_mirr, cent_mirr, pts_right = self.segment_side(t2, res_a0, image_coords_mirr, True)

            print(cent_orig, cent_mirr)
            cent_mirr = cent_mirr[0]
            cent_orig = cent_orig[0]
            self.center_orig = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode')
            self.center_orig.AddControlPointWorld(cent_mirr[0], cent_mirr[1], cent_mirr[2])
            self.center_orig.AddControlPointWorld(cent_orig[0], cent_orig[1], cent_orig[2])

            mesh1 = display_mesh(mesh_orig, "STN right")
            mesh2 = display_mesh(mesh_mirr, "STN left")

            return (mesh1, pts_left), (mesh2, pts_right)

        except Exception as e:
            raise e

    def segment_side(self, t2, center_pred_point, image_coords, mirror=False):
        # load mesh
        mesh = read_mesh(self.resourcePath('nets/3.obj'))
        mm_offset = 2
        # t2 = slicer.util.getNode("t2_normalised")
        image_processor = SlicerImage(t2.GetImageData())

        transform_ras_to_ijk = vtk.vtkMatrix4x4()
        t2.GetIJKToRASMatrix(transform_ras_to_ijk)
        transform_ras_to_ijk.Invert()

        shape_im = compute_image_at_pts(image_processor, image_coords, transform_ras_to_ijk,
                                        (self.shape_label_mask.n_x,
                                         self.shape_label_mask.n_y, self.shape_label_mask.n_z))
        shape_im = self.shape_histogram.apply_normalization(shape_im)
        shape_im = convert_to_tensor(shape_im)

        print("segm start")
        out_pcas = self.shape_predictor(shape_im, False)
        print("segm finished")
        off_cent, result_center = compute_center_offset(center_pred_point, out_pcas, mm_offset)
        shape = compute_shape(self.shape_pca_res[0], out_pcas, result_center)

        res_pts = shape

        print(res_pts.shape, res_pts)
        if mirror:
            shape, result_center = apply_mirror(shape, result_center)

        return change_mesh(mesh, shape), result_center, res_pts


def apply_mirror(shape, result_center):
    shape = shape * [-1, 1, 1]
    result_center = result_center * [-1, 1, 1]
    return shape, result_center


def change_mesh(mesh, ch_pts):
    polys = mesh.GetPolys()
    pts = mesh.GetPoints()
    for i in range(pts.GetNumberOfPoints()):
        pts.SetPoint(i, ch_pts[i, 0], ch_pts[i, 1], ch_pts[i, 2])

    mesh.SetPoints(pts)
    return mesh


def compute_shape(pca_transform, out_pcas, result_center):
    pcas = out_pcas[:, :-3]
    shape = pca_transform.inverse_transform(pcas.detach().numpy())[0]
    shape = np.reshape(shape, (int(shape.shape[0] / 3), 3)) + result_center
    return shape


def compute_center_offset(center_pred_point, out_pcas, mm_offset):
    off_cent = (out_pcas[:, -3:] * 2 * mm_offset) - mm_offset  # reshape center offset
    result_center = center_pred_point + off_cent.detach().numpy()
    return off_cent, result_center


def convert_to_tensor(shape_im):
    shape_im = torch.from_numpy(np.expand_dims(np.expand_dims(shape_im, axis=0), axis=0)).type(torch.float32)
    return shape_im


def compute_image_at_pts(image_processor, image_coords, transform_ras_to_ijk, result_shape):
    shape_im = image_processor.compute_image_at_pts(points=image_coords,
                                                    transform_ras_to_ijk=transform_ras_to_ijk)
    shape_im = np.reshape(shape_im, result_shape)
    return shape_im


def display_mesh(mesh, node_name):
    modelNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode')
    modelNode.SetAndObservePolyData(mesh)
    modelNode.SetDisplayVisibility(True)
    modelNode.SetName(node_name)

    displayNode = modelNode.GetDisplayNode()
    if displayNode is None:
        displayNode: vtkMRMLModelDisplayNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLModelDisplayNode")
        slicer.mrmlScene.AddNode(displayNode)
        modelNode.SetAndObserveDisplayNodeID(displayNode.GetID())
    displayNode.SetScalarVisibility(1)
    displayNode.SetVisibility3D(1)
    displayNode.SetVisibility2D(1)
    displayNode.SetOpacity(0.3)
    displayNode.Modified()
    return modelNode


#
# STNSegmenterTest
#


class STNSegmenterTest(ScriptedLoadableModuleTest):
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
        self.test_STNSegmenter1()

    def test_STNSegmenter1(self):
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
        inputVolume = SampleData.downloadSample("STNSegmenter1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = STNSegmenterLogic()

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

    def segment_side(self, t2, center_pred_point, image_coords, mirror=False):
        # load mesh
        mesh = read_mesh(self.resourcePath('nets/3.obj'))
        mm_offset = 2
        # t2 = slicer.util.getNode("t2_normalised")
        image_processor = SlicerImage(t2.GetImageData())

        transform_ras_to_ijk = vtk.vtkMatrix4x4()
        t2.GetIJKToRASMatrix(transform_ras_to_ijk)
        transform_ras_to_ijk.Invert()

        shape_im = compute_image_at_pts(image_processor, image_coords, transform_ras_to_ijk,
                                        (self.shape_label_mask.n_x,
                                         self.shape_label_mask.n_y, self.shape_label_mask.n_z))
        shape_im = self.shape_histogram.apply_normalization(shape_im)
        shape_im = convert_to_tensor(shape_im)

        print("segm start")
        out_pcas = self.shape_predictor(shape_im, False)
        print("segm finished")
        off_cent, result_center = compute_center_offset(center_pred_point, out_pcas, mm_offset)
        shape = compute_shape(self.shape_pca_res[0], out_pcas, result_center)

        res_pts = shape

        print(res_pts.shape, res_pts)
        if mirror:
            shape, result_center = apply_mirror(shape, result_center)

        return change_mesh(mesh, shape), result_center, res_pts
