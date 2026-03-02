import shlex
import logging
import os
import pickle
import platform
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated, Optional, Tuple

import slicer
from MRMLCorePython import vtkMRMLVolumeArchetypeStorageNode, vtkMRMLTransformNode, vtkMRMLModelNode, \
    vtkMRMLModelDisplayNode



try:
    from dbs_image_utils.mask import SubcorticalMask
except ImportError:
    slicer.util.pip_install('dbs-pure-lib')
    from dbs_image_utils.mask import SubcorticalMask
from dbs_image_utils.nets import CenterDetector, CenterAndPCANet


def _install_antspyx() -> None:
    if (
        platform.system() == "Linux"
        and sys.version_info[:2] == (3, 12)
        and platform.machine().lower() in {"x86_64", "amd64"}
    ):
        import urllib.request

        download_url = "https://app.box.com/shared/static/mu1gy26t80oopbtv3mndl5yveb6s4431.whl"
        whl_filename = "antspyx-0.6.2-cp312-cp312-linux_x86_64.whl"
        whl_path = Path(tempfile.gettempdir()) / whl_filename

        logging.info("Downloading antspyx wheel from %s", download_url)
        urllib.request.urlretrieve(download_url, str(whl_path))
        logging.info("Downloaded antspyx wheel to %s", whl_path)

        try:
            slicer.util.pip_install(str(whl_path))
        finally:
            try:
                whl_path.unlink()
            except OSError:
                pass
        return

    slicer.util.pip_install("antspyx")


try:
    import ants
    import antstorch
except ImportError:
    _install_antspyx()
    slicer.util.pip_install('git+https://github.com/ANTsX/ANTsTorch.git')
    import antstorch
    import ants

import numpy as np
import qt

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


def _looks_like_git_lfs_pointer(file_path: Path) -> bool:
    if not file_path.exists() or file_path.stat().st_size > 1024:
        return False
    with open(file_path, "rb") as f:
        head = f.read(256)
    return b"git-lfs.github.com/spec" in head


def _load_torch_state_dict(
        file_path: Path,
        download_url: Optional[str] = None,
        model_name: str = "model") -> dict:
    def _download_weights() -> None:
        if not download_url:
            raise RuntimeError(
                f"Cannot auto-download weights for {model_name}; missing download URL."
            )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = file_path.with_suffix(file_path.suffix + ".download")
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        success = slicer.util.downloadFile(download_url, str(tmp_path))
        if not success:
            raise RuntimeError(f"Failed to download weights for {model_name} from {download_url}")
        if not tmp_path.exists() or tmp_path.stat().st_size == 0:
            raise RuntimeError(f"Downloaded empty weights file for {model_name} from {download_url}")
        tmp_path.replace(file_path)

    if not file_path.exists() or _looks_like_git_lfs_pointer(file_path):
        _download_weights()

    try:
        return torch.load(str(file_path), map_location=torch.device('cpu'))
    except Exception as first_error:
        if not download_url:
            raise RuntimeError(
                f"Failed loading {model_name} weights from {file_path}: {first_error}"
            ) from first_error
        logging.warning("Failed loading %s weights from %s, re-downloading", model_name, file_path)
        try:
            if file_path.exists():
                file_path.unlink()
        except OSError:
            pass
        _download_weights()
        try:
            return torch.load(str(file_path), map_location=torch.device('cpu'))
        except Exception as second_error:
            raise RuntimeError(
                f"Failed loading {model_name} weights after re-download: {second_error}"
            ) from second_error


def _install_segm_runtime_dependencies() -> None:
    slicer.util.pip_install("dbs-pure-lib")
    slicer.util.pip_install("nibabel")
    slicer.util.pip_install("intensity-normalization")
    slicer.util.pip_install("git+https://github.com/ANTsX/ANTsTorch.git")


def _import_segm_support():
    last_error = None
    for package_name in ("segm_lib", "Lib"):
        try:
            slicer_preprocessing = __import__(
                f"{package_name}.slicer_preprocessing",
                fromlist=["slicer_preprocessing"],
            )
            image_utils = __import__(
                f"{package_name}.image_utils",
                fromlist=["SlicerImage"],
            )
            return slicer_preprocessing, image_utils.SlicerImage
        except ImportError as exc:
            last_error = exc

    _install_segm_runtime_dependencies()

    for package_name in ("segm_lib", "Lib"):
        try:
            slicer_preprocessing = __import__(
                f"{package_name}.slicer_preprocessing",
                fromlist=["slicer_preprocessing"],
            )
            image_utils = __import__(
                f"{package_name}.image_utils",
                fromlist=["SlicerImage"],
            )
            return slicer_preprocessing, image_utils.SlicerImage
        except ImportError as exc:
            last_error = exc

    raise last_error




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

    structuralVolume - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """

    structuralVolume: vtkMRMLScalarVolumeNode
    t2Volume: vtkMRMLScalarVolumeNode
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
        self.t1_node: Optional[vtkMRMLScalarVolumeNode] = None
        self.t2_node: Optional[vtkMRMLScalarVolumeNode] = None
        self.transform_node: Optional[vtkMRMLTransformNode] = None
        self.wm_seg_done = False
        self.intensity_normalisation_done = False
        self.selectedMNIRegistrationMethod = "Rigid"

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

        self._create_temp_folder()
        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        # Keep setup resilient: if initialization fails, handlers stay connected and we show
        # a concrete error on first action instead of appearing unresponsive.
        self._ensureLogic(show_errors=False)
        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.preprocessingButton.connect("clicked(bool)", self.onApplyPreprocessing)
        self.ui.betButton.connect("clicked(bool)", self.brain_extraction)
        self.ui.wmSegmentationButton.connect("clicked(bool)", self.onApplyWMSeg)
        self.ui.wmIntensityNormButton.connect("clicked(bool)", self.onApplyIntensity)
        self.ui.twoStepCoregistrationButton.connect("clicked(bool)", self.onTwoStepCoregistration)
        self.ui.segmentationButton.connect("clicked(bool)", self.onSegmentationButtonClicked)
        self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)",
                                      lambda node, name="t1_node": self.onVolumeSelect(node, name))
        self.ui.t2inputSelector.connect("currentNodeChanged(vtkMRMLNode*)",
                                        lambda node, name="t2_node": self.onVolumeSelect(node, name))

        self.ui.twoStepCoregistrationDropdown.connect("currentTextChanged(QString)", self.onMNIRegistrationMethodChanged)
        dropdown_value = self.ui.twoStepCoregistrationDropdown.currentText
        if dropdown_value:
            self.selectedMNIRegistrationMethod = str(dropdown_value)

        # Make sure parameter node is initialized (needed for module reload)
        if self.logic:
            self.initializeParameterNode()
        # Populate cached node references if selectors already have a node
        self.onVolumeSelect(self.ui.inputSelector.currentNode(), "t1_node")
        self.onVolumeSelect(self.ui.t2inputSelector.currentNode(), "t2_node")

    def onMNIRegistrationMethodChanged(self, method: str):
        """
        Update the selected registration method when the dropdown value changes.
        """
        self.selectedMNIRegistrationMethod = method
        print(f"Selected registration method: {self.selectedMNIRegistrationMethod}")


    def onVolumeSelect(self, x: vtkMRMLScalarVolumeNode, name):
        # get storage node of x
        if x is not None:
            setattr(self, name, x)
        pass

    def _create_temp_folder(self):
        self.temp_workdir = tempfile.TemporaryDirectory()
        print(self.temp_workdir.name)

    def _ensureLogic(self, show_errors=True) -> bool:
        if self.logic:
            return True
        try:
            self.logic = STNSegmenterLogic()
            return True
        except Exception as exc:
            logging.exception("Failed to initialize STNSegmenter logic")
            self.logic = None
            if show_errors:
                slicer.util.errorDisplay(f"Failed to initialize STNSegmenter logic:\n{exc}")
            return False

    def _requireVolume(self, volume_node: Optional[vtkMRMLScalarVolumeNode], volume_name: str) -> bool:
        if volume_node is not None:
            return True
        slicer.util.errorDisplay(f"Select a {volume_name} volume first.")
        return False

    def onApplyPreprocessing(self) -> None:
        if not self._ensureLogic():
            return
        if not self._requireVolume(self.t1_node, "T1"):
            return
        if not self._requireVolume(self.t2_node, "T2"):
            return
        with slicer.util.tryWithErrorDisplay(_("Failed to coregister T2 to structural image."), waitCursor=True):
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
            self.ui.t2inputSelector.setCurrentNode(t2_node)
            self.onVolumeSelect(t2_node, "t2_node")
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
        t1_path = Path(self.temp_workdir.name) / "t1.nii.gz"
        if not t1_path.exists():
            slicer.util.errorDisplay("Run Brain Extraction first to generate a preprocessed T1 image.")
            return
        with slicer.util.tryWithErrorDisplay(_("Failed to compute white matter segmentation."), waitCursor=True):
            print("start on onApplyWMSeg")
            slicer_preprocessing, unused_slicer_image = _import_segm_support()

            slicer_preprocessing.wm_segmentation(t1=str(t1_path),
                                                     out_folder=self.temp_workdir.name)
            self.wm_seg_done = True
            print("fin on appl")

    def onApplyIntensity(self):
        if not self._ensureLogic():
            return
        if not self._requireVolume(self.t2_node, "T2"):
            return
        if not self.wm_seg_done:
            slicer.util.errorDisplay("Run WM segmentation before intensity normalization.")
            return
        with slicer.util.tryWithErrorDisplay(_("Failed to normalize T2 intensity."), waitCursor=True):
            print("test onApplyIntensity")
            t2_storage = check_storage_node(self.t2_node, self.temp_workdir).GetFileName()
            print(t2_storage)
            self.logic.intensity_normalisation(self.temp_workdir.name, t2_file_name=t2_storage)
            self.intensity_normalisation_done = True
            t2_node = loadNiiImage(str(Path(self.temp_workdir.name) / "t2_normalised.nii.gz"))
            self.ui.t2inputSelector.setCurrentNode(t2_node)
            self.onVolumeSelect(t2_node, "t2_node")
            self.popup_window()

    def onTwoStepCoregistration(self):
        if not self._ensureLogic():
            return
        if not self._requireVolume(self.t2_node, "T2"):
            return
        with slicer.util.tryWithErrorDisplay(_("Failed to coregister to MNI."), waitCursor=True):
            print("test onTwoStepCoregustration " + self.selectedMNIRegistrationMethod)
            self.transform_node = self.logic.two_step_coregistration(
                self.t2_node,
                self.temp_workdir.name,
                method=self.selectedMNIRegistrationMethod,
            )
            self.transform_node.SetName("to_mni")

    def apply_normalization(self, shape_im):
        shape_im = self.shape_histogram.apply_normalization(shape_im)
        return shape_im

    def brain_extraction(self):
        if not self._ensureLogic():
            return
        if not self._requireVolume(self.t1_node, "T1"):
            return
        with slicer.util.tryWithErrorDisplay(_("Failed to run brain extraction."), waitCursor=True):
            self.logic.brain_extraction(check_storage_node(self.t1_node, self.temp_workdir), self.temp_workdir.name)
            t1_node = loadNiiImage(str(Path(self.temp_workdir.name) / "t1.nii.gz"))
            self.ui.inputSelector.setCurrentNode(t1_node)
            self.onVolumeSelect(t1_node, "t1_node")

    def onSegmentationButtonClicked(self):
        if not self._ensureLogic():
            return
        if not self._requireVolume(self.t2_node, "T2"):
            return
        if self.transform_node is None:
            slicer.util.errorDisplay("Run 'Coregistration to MNI' before STN segmentation.")
            return
        with slicer.util.tryWithErrorDisplay(_("Failed to segment STNs."), waitCursor=True):
            print("Starting segmentation")

            left, right = self.logic.segmentSTNs(self.t2_node)

            # invert tranform node
            inverted_transform = slicer.mrmlScene.CopyNode(self.transform_node)
            inverted_transform.SetName("to_mni_inverted")
            inverted_transform.Inverse()

            slicer.mrmlScene.AddNode(inverted_transform)
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
        if not self._ensureLogic(show_errors=False):
            return

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.structuralVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.structuralVolume = firstVolumeNode

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
        if self._parameterNode and self._parameterNode.structuralVolume and self._parameterNode.thresholdedVolume:
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
        cd_state_dict = _load_torch_state_dict(
            Path(self.resourcePath('nets/cent_pred.pt')),
            model_name="center predictor",
        )
        net.load_state_dict(cd_state_dict)
        self.center_detector = net

        self.shape_pca_res = _read_pickle(self.resourcePath('nets/shape_pcas.pkl'))

        self.shape_label_mask: SubcorticalMask = _read_pickle(self.resourcePath('nets/stn_shape_mask.pkl'))

        # load segmentation model
        self.shape_histogram = _read_pickle(self.resourcePath('nets/shape_hist.pkl'))
        net = CenterAndPCANet(self.shape_pca_res[1])
        shape_weights_path = Path(self.resourcePath('nets/shp_pred.pt'))
        cd_state_dict = _load_torch_state_dict(
            shape_weights_path,
            download_url="https://github.com/IVarha/SlicerDBSCoalignment/releases/download/0.0.1/shp_pred.pt",
            model_name="shape predictor",
        )
        net.load_state_dict(cd_state_dict)
        self.shape_predictor = net


    def getParameterNode(self):
        return STNSegmenterParameterNode(super().getParameterNode())

    def process(self,
                structuralVolume: vtkMRMLScalarVolumeNode,
                t2Volume: vtkMRMLScalarVolumeNode,
                imageThreshold: float,
                invert: bool = False,
                showResult: bool = True) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param structuralVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """

        if not structuralVolume or not t2Volume:
            raise ValueError("Input or T2 volume is invalid")

        import time

        startTime = time.time()
        logging.info("Processing started")
        right, left = None,None
        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        # create temp workdir
        try:
            temp_workdir = tempfile.TemporaryDirectory()
            sn1 = STNSegmenter.check_storage_node(structuralVolume, temp_workdir)
            sn2 = STNSegmenter.check_storage_node(t2Volume, temp_workdir)

            # register t2 to t1
            self.coregistration_t2_t1(sn1
                                       , t2=sn2,
                                       out_name=str(Path(temp_workdir.name) / "coreg_t2.nii.gz"))
            t2_node = STNSegmenter.loadNiiImage(str(Path(temp_workdir.name) / "coreg_t2.nii.gz"))
            # brain extraction

            self.brain_extraction(sn1, temp_workdir.name)

            t1_node = STNSegmenter.loadNiiImage(str(Path(temp_workdir.name) / "t1.nii.gz"))
            print("Brain extracted")
            # # wm segmentation
            self.wm_segmentation(str(Path(temp_workdir.name) / "t1.nii.gz"), temp_workdir.name)
            print("WM segmented")
            # # intensity normalisation
            self.intensity_normalisation(temp_workdir.name)
            t2_node = STNSegmenter.loadNiiImage(str(Path(temp_workdir.name) / "t2_normalised.nii.gz"))
            print("Coregistering T2 to T1")
            transform_node = self.two_step_coregistration(t2_node, temp_workdir.name)
            transform_node.SetName("to_mni")

            (right, _), (left, _) = self.segmentSTNs(t2_node)
            print("STNs segmented")

            #
        finally:
            self.processing_folder.cleanup()
            self.processing_folder = None



        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime - startTime:.2f} seconds")
        return right,left
    def brain_extraction(self, t1: vtkMRMLVolumeArchetypeStorageNode, temp_dir_path) -> None:
        image_name = t1.GetFileName()
        mask_filename = str(Path(temp_dir_path) / "t1_mask.nii.gz")


        img = ants.image_read(image_name)

        try:
            # Use a per-run cache to avoid corrupt global ANTsTorch caches.
            try:
                from antstorch.utilities.get_antstorch_data import set_antstorch_cache_directory

                cache_dir = Path(temp_dir_path) / "antstorch_cache"
                cache_dir.mkdir(parents=True, exist_ok=True)
                set_antstorch_cache_directory(str(cache_dir))
            except Exception:
                pass
            mask = antstorch.brain_extraction(img, "t1") > 0.8
        except EOFError as e:
            # Corrupted cached weights can trigger EOFError during torch.load.
            try:
                import os
                import shutil
                from antstorch.utilities.get_pretrained_network import get_pretrained_network

                # Clear the per-run cache if present.
                try:
                    cache_dir = Path(temp_dir_path) / "antstorch_cache"
                    if cache_dir.exists():
                        shutil.rmtree(cache_dir, ignore_errors=True)
                except Exception:
                    pass

                weights_file = get_pretrained_network(
                    "brainExtractionRobustT1_pytorch",
                    target_file_name="brainExtractionRobustT1_pytorch.pt",
                )
                if os.path.exists(weights_file):
                    os.remove(weights_file)
            except Exception:
                raise RuntimeError(
                    "ANTsTorch brain extraction failed while handling a corrupted weights cache. "
                    "Delete the cached weights in %USERPROFILE%\\.antstorch and retry."
                ) from e

            # Retry once after clearing the cached weights.
            mask = antstorch.brain_extraction(img, "t1") > 0.8
        masked_image = img * mask
        ants.image_write(mask, mask_filename)
        ants.image_write(masked_image, str(Path(temp_dir_path) / "t1.nii.gz"))


        print("FINISHED EXTRACTOR")
        # cmd = [sys.executable, self.resourcePath("py/bet.py"), str(image_name), mask_filename, str(Path(temp_dir_path) / "t1.nii.gz")]
        # print(cmd)
        # subprocess.call(cmd, shell=True)

    def coregistration_t2_t1(self, t1: vtkMRMLVolumeArchetypeStorageNode, t2: vtkMRMLVolumeArchetypeStorageNode,
                             out_name: str) -> None:
        slicer_preprocessing, unused_slicer_image = _import_segm_support()

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
        slicer_preprocessing, unused_slicer_image = _import_segm_support()

        slicer_preprocessing.wm_segmentation(t1, out_folder)

    def _get_elastix_executable(self):
        import Elastix
        lgc= Elastix.ElastixLogic()

        res = Path(lgc.getElastixBinDir())/ lgc.elastixFilename

        return str(res)

    def intensity_normalisation(self, out_folder: str, t2_file_name: str) -> None:
        slicer_preprocessing, unused_slicer_image = _import_segm_support()
        slicer_preprocessing.intensity_normalisation(out_folder,t2_file_name)

    def two_step_coregistration(self, node_to_transform, workdir: str, method = "Rigid") -> vtkMRMLTransformNode:
        # Define paths
        import ants
        # Define paths
        mni = self.resourcePath('MNI/MNI152_T1_1mm_brain.nii.gz')  # Reference image (fixed)
        struct_image = str(Path(workdir) / "t1.nii.gz")  # Moving image
        output_transform_prefix = str(Path(workdir) / "transform")  # Prefix for output transforms

        # Load fixed and moving images using ANTs
        fixed_image = ants.image_read(mni)
        moving_image = ants.image_read(struct_image)

        # Perform rigid registration
        registration = ants.registration(
            fixed=fixed_image,
            moving=moving_image,
            type_of_transform=method,  # Use rigid transformation
            outprefix=output_transform_prefix
        )

        print("fwdtransfs : ", registration['fwdtransforms'])

        len_reg = len(registration['fwdtransforms'])
        transform_file = registration['fwdtransforms'][0]  # Path to the forward transform file

        transform_node = slicer.util.loadTransform(transform_file)
        if len_reg > 1:
            transform_file_next = None
            for i in range(1, len_reg):
                transform_file_next = registration['fwdtransforms'][i]
                transform_node_next = slicer.util.loadTransform(transform_file_next)
                transform_node.SetAndObserveTransformNodeID(transform_node_next.GetID())
                transform_node.HardenTransform()




        # # Get the forward transform file from the registration results
        # transform_file = registration['fwdtransforms'][0]  # Path to the forward transform file
        #
        # transform_node = slicer.util.loadTransform(transform_file)

        # Apply the transform to the input node
        node_to_transform.SetAndObserveTransformNodeID(transform_node.GetID())
        node_to_transform.HardenTransform()

        # Return the transform node
        return transform_node

    def segmentSTNs(self, t2_node) -> Tuple[MESH_results, MESH_results]:
        mm_offset = 2
        print("Starting segmentation")
        unused_slicer_preprocessing, SlicerImage = _import_segm_support()

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
        unused_slicer_preprocessing, SlicerImage = _import_segm_support()

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
    displayNode.SetColor(207 / 255., 75/255., 75 / 255.)
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
        structuralVolume = SampleData.downloadSample("STNSegmenter1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = structuralVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = STNSegmenterLogic()

        # Test algorithm with non-inverted threshold
        logic.process(structuralVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(structuralVolume, outputVolume, threshold, False)
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

