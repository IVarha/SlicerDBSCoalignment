import logging
import os
import pickle
import shutil
import subprocess
from typing import Annotated, Optional
import sys
import numpy as np
import qt
from Lib.mer_support import EntryTarget, Point, cross_generation_mni, ElectrodeRecord, ElectrodeArray, \
    optimisation_criterion, clasify_mers
from Lib.visualisiation import LeadORLogic
import torch
import vtk
import tempfile
import slicer
from sklearn.preprocessing import MinMaxScaler
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)
from mer_lib.data import MER_data
import nibabel as nib
from brainextractor import BrainExtractor
import fsl.data.image as fim
import fsl.transform.flirt as fl
from pathlib import Path
from Lib import slicer_preprocessing
from Lib.image_utils import SlicerImage
from Lib.utils_file import get_images_in_folder
from slicer import vtkMRMLScalarVolumeNode
from dbs_image_utils.mask import SubcorticalMask
import sitkUtils
from dbs_image_utils.histogram_standartisation import HistogramStandartisation
from dbs_image_utils.nets import CenterDetector, CenterAndPCANet, TransformerShiftPredictor, TransformerClassifier
import mer_lib.processor as proc
import mer_lib.artefact_detection as ad
import mer_lib.feature_extraction as fe
import mer_lib.data as mer_dat


def nrms_normalisation(data: MER_data):
    """
    across all trajectories together:
        NRMS = 1 + (NRMS1-1) / (p95(NRMS1)-1)
    """

    extracted_features = data.extracted_features
    percentile_95 = np.percentile(extracted_features, 95)
    data.extracted_features = 1 + (extracted_features - 1) / (percentile_95 - 1)

    return data


def get_flirt_transformation_matrix(mat_file, src_file, dest_file, from_, to):
    """
    Get the transformation matrix using FLIRT.

    Args:
        mat_file (str): Path to the FLIRT transformation matrix file.
        src_file (str): Path to the source image file.
        dest_file (str): Path to the destination image file.
        from_ (str): Source image space.
        to (str): Destination image space.

    Returns:
        numpy.ndarray: The transformation matrix.

    """
    im_src = fim.Image(src_file, loadData=False)
    im_dest = fim.Image(dest_file, loadData=False)
    forward_transf_fsl = fl.readFlirt(mat_file)
    return fl.fromFlirt(forward_transf_fsl, im_src, im_dest, from_, to)


def get_control_points(markups_node):
    """
    Retrieves the world positions of all control points in a markups node.

    Args:
        markups_node: The markups node from which to retrieve the control points.

    Returns:
        A list of world positions of all control points in the markups node.
    """
    points = []
    for i in range(markups_node.GetNumberOfControlPoints()):
        points.append(markups_node.GetNthControlPointPositionWorld(i))
    return points


def get_transform_matrix(transform_node):
    """
    Get the transformation matrix of a given transform node.

    Parameters:
    transform_node (vtk.vtkTransformNode): The transform node to get the matrix from.

    Returns:
    vtk.vtkMatrix4x4: The transformation matrix.
    """
    matrix = vtk.vtkMatrix4x4()
    transform_node.GetMatrixTransformToParent(matrix)
    return matrix


def convert_to_numpy_array(matrix):
    numpy_array = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            numpy_array[i, j] = matrix.GetElement(i, j)
    return numpy_array


def apply_transformation(entry_target, transformation_matrix):
    return cross_generation_mni(ent_tg_native=entry_target, to_mni=transformation_matrix)


def read_mer_data(file_path, pattern):
    return MER_data.read_ontario_data(str(Path(file_path).parent), pattern)


def process_mer_data(mer_data):
    """
    Process the MER data using a series of predefined steps.

    Args:
        mer_data (list): The MER data to be processed.

    Returns:
        list: The processed MER data.
    """
    def min_max_scaler(data: MER_data):
        extracted_features = data.extracted_features
        data.extracted_features = (extracted_features - extracted_features.min())/(extracted_features.max() - extracted_features.min())

        return data

    runner = proc.Processor()
    runner.set_processes([ad.covariance_method, fe.nrms_calculation, nrms_normalisation,min_max_scaler])
    mer_data = runner.run(mer_data)
    return mer_data


def extract_electrode_records(entry_target, mer_data, transformation_matrix):
    """
    Extracts electrode records from an array of MER data.

    Args:
        entry_target (str): The target entry for electrode extraction.
        mer_data (array): The array of MER data.
        transformation_matrix (array): The transformation matrix for coordinate transformation.

    Returns:
        list: A list of extracted electrode records.
    """
    return ElectrodeRecord.extract_electrode_records_from_array(entry_target, mer_data,
                                                                transformation=transformation_matrix)


def read_mesh(file_name):
    mesh = vtk.vtkOBJReader()
    mesh.SetFileName(file_name)
    mesh.Update()
    mesh = mesh.GetOutput()
    return mesh


def optimise_mer_signal(mer_data, origin_shift: np.ndarray, mesh):
    """
    optimise the signal of the mer data to get the best overlap with the mesh
    input: mer_data: List[ElectrodeRecord]

    """
    # convert mer_data to torch tensor
    x, mer_label = ElectrodeRecord.electrode_list_to_array(mer_data)


    mer_data = torch.from_numpy(x).type(torch.float32)

    mer_label = torch.from_numpy(mer_label).type(torch.int)

    optimise_f = lambda x: optimisation_criterion(mer_data, mer_label, shift=x, mesh=mesh) + torch.linalg.norm(x)

    x = torch.from_numpy(origin_shift).requires_grad_(True)

    print("Origin estimated value of shift:", x)
    print("Origin estimated value of the function:", optimise_f(x).item())
    print("original estimated distance", torch.linalg.norm(x))
    # Define the optimizer
    optimizer = torch.optim.SGD([x], lr=0.004)

    # Optimization loop
    for i in range(100):
        optimizer.zero_grad()  # Zero out the gradients
        output = optimise_f(x)  # Compute the function value
        output.backward()  # Compute gradients
        optimizer.step()  # Update parameters

    print("Optimized value of x:", x)
    print("final estimated distance", torch.linalg.norm(x))
    print("Optimized value of the function:", optimise_f(x).item())

    return x.detach().numpy()


#
# load_nifty
#

class load_nifty(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "DBS Subcortical electrode localisation"  # TODO: make this more human readable by adding spaces
        self.parent.categories = [
            "Examples"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = [
            "John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#load_nifty">module documentation</a>.
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#

def registerSampleData():
    """
    Add data sets to Sample Data module.
    """
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData
    iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # load_nifty1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='load_nifty',
        sampleName='load_nifty1',
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, 'load_nifty1.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames='load_nifty1.nrrd',
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums='SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
        # This node name will be used when the data set is loaded
        nodeNames='load_nifty1'
    )

    # load_nifty2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='load_nifty',
        sampleName='load_nifty2',
        thumbnailFileName=os.path.join(iconsPath, 'load_nifty2.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames='load_nifty2.nrrd',
        checksums='SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
        # This node name will be used when the data set is loaded
        nodeNames='load_nifty2'
    )


#
# load_niftyParameterNode
#

@parameterNodeWrapper
class load_niftyParameterNode:
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
# load_niftyWidget
#

def change_mesh(mesh, ch_pts):
    polys = mesh.GetPolys()
    pts = mesh.GetPoints()
    for i in range(pts.GetNumberOfPoints()):
        pts.SetPoint(i, ch_pts[i, 0], ch_pts[i, 1], ch_pts[i, 2])

    mesh.SetPoints(pts)
    return mesh


class load_niftyWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Widget class for loading and processing NIFTY files.

    This class inherits from ScriptedLoadableModuleWidget and VTKObservationMixin.
    It provides a graphical user interface for loading and processing NIFTY files.

    Attributes:
        mesh1 (None): Placeholder for mesh1.
        center_orig (None): Placeholder for center_orig.
        mesh2 (None): Placeholder for mesh2.
        shape_predictor (None): Placeholder for shape_predictor.
        center_detector_scaller (None): Placeholder for center_detector_scaller.
        cent_det_hist (None): Placeholder for cent_det_hist.
        det_mask (None): Placeholder for det_mask.
        intensity_normalisation_done (bool): Flag indicating if intensity normalization is done.
        wm_seg_done (bool): Flag indicating if white matter segmentation is done.
        logic (None): Placeholder for logic class.
        _parameterNode (None): Placeholder for parameter node.
        _parameterNodeGuiTag (None): Placeholder for parameter node GUI tag.
        temp_workdir (TemporaryDirectory): Temporary working directory.
        shape_pca_res (None): Placeholder for shape_pca_res.
        shape_label_mask (None): Placeholder for shape_label_mask.
        shape_histogram (None): Placeholder for shape_histogram.
        center_detector (None): Placeholder for center_detector.
    """

    def __init__(self, parent=None) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.mesh1 = None
        self.center_orig = None
        self.mesh2 = None
        self.shape_predictor = None
        self.center_detector_scaller = None
        self.cent_det_hist = None
        self.det_mask = None
        self.intensity_normalisation_done = False
        self.wm_seg_done = False
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    # Rest of the code...


def convert_to_tensor(shape_im):
    shape_im = torch.from_numpy(np.expand_dims(np.expand_dims(shape_im, axis=0), axis=0)).type(torch.float32)
    return shape_im


def compute_center_offset(center_pred_point, out_pcas, mm_offset):
    off_cent = (out_pcas[:, -3:] * 2 * mm_offset) - mm_offset  # reshape center offset
    result_center = center_pred_point + off_cent.detach().numpy()
    return off_cent, result_center


def reshape_shape(shape):
    shape = np.reshape(shape, (-1, 1))
    return shape


def apply_mirror(shape, result_center):
    shape = shape * [-1, 1, 1]
    result_center = result_center * [-1, 1, 1]
    return shape, result_center


class load_niftyWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.mer_classification_net = None
        self.processing_folder = None
        self.mesh1 = None
        self.center_orig = None
        self.mesh2 = None
        self.shape_predictor = None
        self.center_detector_scaller = None
        self.cent_det_hist = None
        self.det_mask = None
        self.intensity_normalisation_done = False
        self.wm_seg_done = False
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/load_nifty.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = load_niftyLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        self.ui.t2InputSelector.connect('textActivated(QString)', lambda x: self.on_chage_load_image("t2", x))
        self.ui.structuralInputSelector.connect('textActivated(QString)', lambda x: self.on_chage_load_image("t1", x))
        # Buttons
        # self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)
        self.ui.preprocessingButton.clicked.connect(self.onApplyPreprocessing)
        self.ui.betButton.clicked.connect(self.brain_extraction)
        self.ui.wmSegmentationButton.clicked.connect(self.onApplyWMSeg)
        self.ui.wmIntensityNormButton.clicked.connect(self.onApplyIntensity)
        self.ui.twoStepCoregistrationButton.clicked.connect(self.onTwoStepCoregistration)
        self.ui.segmentationButton.clicked.connect(self.onSegmentationButtonClicked)
        self.ui.inputPathSelector.connect('currentPathChanged(QString)', self.onInputFolderSelect)
        self.ui.merButton.clicked.connect(self.onMerRunButtonClicked)
        self.ui.merFinalButton.clicked.connect(self.on_calculate_shift)
        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

        # create temp workdir
        self._create_temp_folder()
        # self.lead_or_bind()
        # initialise variables
        print(self.resourcePath('nets/cent_detect_hist.pkl'))
        self.cent_det_hist = self._read_pickle(self.resourcePath('nets/cent_detect_hist.pkl'))

        self.det_mask: SubcorticalMask = self._read_pickle(self.resourcePath('nets/detect_mask.pkl'))
        print(self.det_mask.get_coords_3d()[self.det_mask.n_x // 2, self.det_mask.n_y // 2, self.det_mask.n_z // 2, :])

        def _compute_min_max_scaler(pt_min, pt_max):
            a = MinMaxScaler()
            a.fit([pt_min, pt_max])
            return a

        # load center detector
        self.center_detector_scaller = _compute_min_max_scaler(self.det_mask.min_p, self.det_mask.max_p)
        net = CenterDetector().to('cpu')
        cd_state_dict = torch.load(self.resourcePath('nets/cent_pred.pt'), map_location=torch.device('cpu'))
        net.load_state_dict(cd_state_dict)
        self.center_detector = net

        self.shape_pca_res = self._read_pickle(self.resourcePath('nets/shape_pcas.pkl'))

        self.shape_label_mask: SubcorticalMask = self._read_pickle(self.resourcePath('nets/stn_shape_mask.pkl'))

        # load segmentation model
        self.shape_histogram = self._read_pickle(self.resourcePath('nets/shape_hist.pkl'))
        net = CenterAndPCANet(self.shape_pca_res[1])
        cd_state_dict = torch.load(self.resourcePath('nets/shp_pred.pt'), map_location=torch.device('cpu'))
        net.load_state_dict(cd_state_dict)
        self.shape_predictor = net

        self.net_shift = TransformerShiftPredictor(4, 32, 1, 1, 10)
        cd_state_dict = torch.load(self.resourcePath('nets/net_transformer.pt'), map_location=torch.device('cpu'))
        self.net_shift.load_state_dict(cd_state_dict)

        self.mer_transforms = self._read_pickle(self.resourcePath('nets/mer_pca_mesh.pkl'))

        self.mer_classification_net = TransformerClassifier(4,64,1,1)
        cd_state_dict = torch.load(self.resourcePath('nets/transformer_classifier.pt'), map_location=torch.device('cpu'))
        self.mer_classification_net.load_state_dict(cd_state_dict)


    def _read_pickle(self, filename):
        f = open(filename, 'rb')
        res = pickle.load(f)
        f.close()
        return res

    def on_chage_load_image(self, key, newvalue):
        x = None
        try:
            x = getattr(self, key)
            print(1)
        except:
            print(1.1)
            pass
        if x is not None:
            if x == newvalue:
                return

        if not hasattr(self, key):
            #            print(newvalue)
            setattr(self, key, newvalue)
        else:
            print(1)

        print('I am loading ', newvalue, " to ", key)
        print(Path(str(self.temp_workdir.name)) / (key + ".nii.gz"))
        shutil.copy(
            Path(str(self.processing_folder / newvalue)),
            Path(str(self.temp_workdir.name)) / (key + ".nii.gz")
        )

        try:
            node = slicer.util.getNode(key)
            slicer.mrmlScene.RemoveNode(node)

            # node is found
        except:  # node not found
            pass  # todo change this method to get t2
        node = loadNiiImage(str(self.processing_folder / newvalue))
        node.SetName(key)

    def cleanup(self) -> None:
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()
        self._remove_temporary_folder()

    def _create_temp_folder(self):
        self.temp_workdir = tempfile.TemporaryDirectory()
        print(self.temp_workdir.name)

    def _remove_temporary_folder(self):
        self.temp_workdir.cleanup()

    def enter(self) -> None:
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        # if not self._parameterNode.inputVolume:
        #     firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        #     if firstVolumeNode:
        #         self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[load_niftyParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """
        pass

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.thresholdedVolume:
            self.ui.applyButton.toolTip = "Compute output volume"
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = "Select input and output volume nodes"
            self.ui.applyButton.enabled = False

    def onApplyPreprocessing(self) -> None:
        print("start on appl")

        slicer_preprocessing.elastix_registration(ref_image=str(self.processing_folder / self.t1),
                                                  flo_image=str(self.processing_folder / self.t2),
                                                  elastix_parameters=self.resourcePath('elastix/rigid_mri.txt')
                                                  , out_folder=self.temp_workdir.name
                                                  )
        ((Path(self.temp_workdir.name) / "result.0.nii.gz")
         .rename((Path(self.temp_workdir.name) / "coreg_t2.nii.gz")))

        self.popup_window()
        print("fin on appl")

    def onApplyWMSeg(self) -> None:
        print("start on onApplyWMSeg")
        slicer_preprocessing.wm_segmentation(t1=str(Path(self.temp_workdir.name) / "t1.nii.gz"),
                                             out_folder=self.temp_workdir.name)
        self.wm_seg_done = True
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

    def onApplyIntensity(self):
        print("test onApplyIntensity")
        # slicer_preprocessing.intensity_normalisation(self.temp_workdir.name)
        if self.wm_seg_done:
            slicer_preprocessing.intensity_normalisation(self.temp_workdir.name)
            self.intensity_normalisation_done = True
            self.t2_node = slicer.util.getNode('t2')
            slicer.mrmlScene.RemoveNode(self.t2_node)
            t2_node = loadNiiImage(str(Path(self.temp_workdir.name) / "t2_normalised.nii.gz"))
            # t2_node.setName("t2_normalised")
            self.t2_node = t2_node

            pass
        self.popup_window()

    def onTwoStepCoregistration(self):
        print("test onTwoStepCoregustration")
        mni = self.resourcePath('MNI/MNI152_T1_1mm_brain.nii.gz')
        mni_mask = self.resourcePath('MNI/MNI152_T1_1mm_subbr_mask.nii.gz')
        elastix_affine = self.resourcePath('elastix/affine_mri.txt')
        struct_image = str(Path(self.temp_workdir.name) / "t1.nii.gz")
        slicer_preprocessing.elastix_registration(ref_image=mni,
                                                  flo_image=struct_image,
                                                  elastix_parameters=elastix_affine,
                                                  out_folder=self.temp_workdir.name)
        tfm_file = Path(self.temp_workdir.name) / "TransformParameters.0.tfm"
        transfortm_node = slicer.util.loadTransform(tfm_file)
        transfortm_node.SetName("to_mni")
        self.transform_node = transfortm_node
        t2_image = slicer.util.getNode("t2_normalised")
        t2_image.SetAndObserveTransformNodeID(transfortm_node.GetID())
        t2_image.HardenTransform()

    def onApplyButton(self) -> None:
        """
        Run processing when user clicks "Apply" button.
        """
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):
            # Compute output
            self.logic.process(self.ui.inputSelector.currentNode(), self.ui.outputSelector.currentNode(),
                               self.ui.imageThresholdSliderWidget.value, self.ui.invertOutputCheckBox.checked)

            # Compute inverted output (if needed)
            if self.ui.invertedOutputSelector.currentNode():
                # If additional output volume is selected then result with inverted threshold is written there
                self.logic.process(self.ui.inputSelector.currentNode(), self.ui.invertedOutputSelector.currentNode(),
                                   self.ui.imageThresholdSliderWidget.value, not self.ui.invertOutputCheckBox.checked,
                                   showResult=False)

    def onInputFolderSelect(self, new_path) -> None:
        """
        run when input folder selected
        """
        print("new_path", new_path)
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):
            self.processing_folder = Path(new_path)
            ###load folder]
            images = get_images_in_folder(new_path)
            self.ui.structuralInputSelector.clear()
            self.ui.structuralInputSelector.addItems(images)
            # for i in images

            self.ui.t2InputSelector.clear()
            self.ui.t2InputSelector.addItems(images)

            #### load
            pass

    def segment_side(self, center_pred_point, image_coords, mirror=False):
        # load mesh
        mesh = read_mesh(self.resourcePath('nets/3.obj'))
        mm_offset = 2
        t2 = slicer.util.getNode("t2_normalised")
        image_processor = SlicerImage(t2.GetImageData())

        transform_ras_to_ijk = vtk.vtkMatrix4x4()
        t2.GetIJKToRASMatrix(transform_ras_to_ijk)
        transform_ras_to_ijk.Invert()

        shape_im = self.compute_image_at_pts(image_processor, image_coords, transform_ras_to_ijk)
        shape_im = self.apply_normalization(shape_im)
        shape_im = convert_to_tensor(shape_im)

        print("segm start")
        out_pcas = self.shape_predictor(shape_im, False)
        print("segm finished")
        off_cent, result_center = compute_center_offset(center_pred_point, out_pcas, mm_offset)
        shape = self.compute_shape(out_pcas, result_center)

        #shape = reshape_shape(shape)
        res_pts = shape

        print(res_pts.shape,res_pts)
        if mirror:
            shape, result_center = apply_mirror(shape, result_center)

        return change_mesh(mesh, shape), result_center, res_pts

    def compute_image_at_pts(self, image_processor, image_coords, transform_ras_to_ijk):
        shape_im = image_processor.compute_image_at_pts(points=image_coords,
                                                        transform_ras_to_ijk=transform_ras_to_ijk)
        shape_im = np.reshape(shape_im, (self.shape_label_mask.n_x,
                                         self.shape_label_mask.n_y, self.shape_label_mask.n_z))
        return shape_im

    def apply_normalization(self, shape_im):
        shape_im = self.shape_histogram.apply_normalization(shape_im)
        return shape_im

    def compute_shape(self, out_pcas, result_center):
        pcas = out_pcas[:, :-3]
        shape = self.shape_pca_res[0].inverse_transform(pcas.detach().numpy())[0]
        shape = np.reshape(shape, (int(shape.shape[0] / 3), 3)) + result_center
        return shape

    def brain_extraction(self):

        image_t1 = Path(self.temp_workdir.name) / "t1.nii.gz"

        print("FINISHED EXTRACTOR")
        mask_filename = str(Path(self.temp_workdir.name) / "t1_mask.nii.gz")
        cmd = sys.executable + " "
        cmd += self.resourcePath("py/bet.py") + " "
        cmd += str(image_t1) + " " + mask_filename
        subprocess.call(cmd, shell=True)

        self.t1_node = slicer.util.getNode('t1')
        slicer.mrmlScene.RemoveNode(self.t1_node)
        t1_node = loadNiiImage(str(Path(self.temp_workdir.name) / "t1.nii.gz"))
        self.t1_node = t1_node
        # t2_node.setName("t2_normalised")

        pass

        # shape = np.reshape(shape, (-1, 1))

    def onSegmentationButtonClicked(self):
        mm_offset = 2
        print("Starting segmentation")

        t2 = slicer.util.getNode("t2_normalised")
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

            mesh_orig, cent_orig, self.pts_left = self.segment_side(res_a1, image_coords_orig)
            mesh_mirr, cent_mirr, self.pts_right = self.segment_side(res_a0, image_coords_mirr, True)

            print(cent_orig, cent_mirr)
            cent_mirr = cent_mirr[0]
            cent_orig = cent_orig[0]
            self.center_orig = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode')
            self.center_orig.AddControlPointWorld(cent_mirr[0], cent_mirr[1], cent_mirr[2])
            self.center_orig.AddControlPointWorld(cent_orig[0], cent_orig[1], cent_orig[2])

            self.mesh1 = display_mesh(mesh_orig, "Mesh left")
            self.mesh2 = display_mesh(mesh_mirr, "Mesh right")



        except Exception as e:
            raise e
            # print(5)

    def onMerRunButtonClicked(self):
        """
        Function triggered when the "MER Run" button is clicked.
        Applies a transformation to the right and left markups nodes and prints the current paths of the MER files.
        """
        right_markups = slicer.mrmlScene.GetNodeByID(self.ui.rightMarkupSelector.currentNodeID)
        left_markups = slicer.mrmlScene.GetNodeByID(self.ui.leftMarkupSelector.currentNodeID)

        points_right = get_control_points(right_markups)
        points_left = get_control_points(left_markups)

        print(points_right)  # Print the positions of the control points for the right markups node
        print(points_left)  # Print the positions of the control points for the left markups node

        matrix = get_transform_matrix(transform_node=self.transform_node)
        numpy_array = convert_to_numpy_array(matrix)

        et_right = EntryTarget(Point.from_array(points_right[0]), Point.from_array(points_right[1]))
        et_left = EntryTarget(Point.from_array(points_left[0]), Point.from_array(points_left[1]))

        ea_left = apply_transformation(et_left, numpy_array)
        ea_right = apply_transformation(et_right, numpy_array)

        print(ea_right)  # Print the entry and target points for the right MER

        self.mer_left_path = self.ui.LeftMERFileSelector.currentPath
        self.mer_right_path = self.ui.RightMERFileSelector.currentPath

        self.mer_left = read_mer_data(self.mer_left_path, "*run-02*")
        self.mer_right = read_mer_data(self.mer_right_path, "*run-01*")

        self.mer_left = process_mer_data(self.mer_left)
        self.mer_right = process_mer_data(self.mer_right)

        self.right_e_rec = extract_electrode_records(ea_right, self.mer_right, numpy_array)
        self.left_e_rec = extract_electrode_records(ea_left, self.mer_left, numpy_array)

        logic = LeadORLogic()
        i = 0
        for el_name, records in self.right_e_rec.items():
            print(records)
            logic.setUpTrajectory(i, getattr(ea_right, el_name), records, True, "right_" + el_name, 1, 1, 1)

        for el_name, records in self.left_e_rec.items():
            print(records)
            logic.setUpTrajectory(i, getattr(ea_left, el_name), records, True, "left_" + el_name, 1, 1, 1)

        # mirror right pts
        records_right = {}
        for key, val in self.right_e_rec.items():
            records_right[key] = []
            for rec in val:
                records_right[key].append(ElectrodeRecord(rec.location * [-1, 1, 1], rec.record, 0))

        self.right_shift = compute_transformation_from_signal(records_right, self.pts_right,
                                                              self.mer_transforms['min_max_p'],
                                                              self.mer_transforms['pipe'], self.net_shift)
        self.left_shift = compute_transformation_from_signal(self.left_e_rec, self.pts_left,
                                                             self.mer_transforms['min_max_p'],
                                                             self.mer_transforms['pipe'], self.net_shift)
        records_right = clasify_mers(records_right, self.mer_classification_net)
        for key, val in records_right.items():
            for rec in val:
                rec.location = rec.location * [-1, 1, 1]
        self.right_e_rec = records_right
        self.left_e_rec = clasify_mers(self.left_e_rec, self.mer_classification_net)

    def on_calculate_shift(self):

        # compute left shift
        mesh = read_mesh(self.resourcePath('nets/3.obj'))
        mesh = change_mesh(mesh, self.pts_left)

        left_list = []
        for key, val in self.left_e_rec.items():
            for rec in val:
                left_list.append(ElectrodeRecord(rec.location, rec.record, rec.label))
        left_shift = optimise_mer_signal(left_list, self.left_shift, mesh)

        right_list = []
        for key, val in self.right_e_rec.items():
            for rec in val:
                right_list.append(ElectrodeRecord(rec.location * [-1, 1, 1], rec.record, rec.label))

        mesh = change_mesh(mesh, self.pts_right)
        # compute right shift
        right_shift = optimise_mer_signal(right_list, self.left_shift, mesh)
        right_shift = right_shift * [-1, 1, 1]

        generate_transformation_from_shift(left_shift, "left_shift")
        generate_transformation_from_shift(right_shift, "right_shift")

        pass


def compute_transformation_from_signal(records,
                                       mesh_pts,
                                       min_max_pts,
                                       transformer, model):
    """

    """
    transl_scaller = MinMaxScaler().fit([[-10, -10, -10], [10, 10, 10]])  # config tesy
    m2 = np.reshape(mesh_pts,(-1,1))
    m2 = m2.reshape(-1).tolist()

    #print(m2.shape, m2)
    pcas = transformer.transform([m2])
    #print()
    flat_list = [item for sublist in records.values() for item in sublist]
    x, _ = ElectrodeRecord.electrode_list_to_array(flat_list)

    pcas = torch.from_numpy(pcas).type(torch.float32)
    data = torch.from_numpy(x).type(torch.float32)

    data[:, :3] = (data[:, :3] - min_max_pts[0]) / (min_max_pts[1] - min_max_pts[0])

    res = model(data, pcas)
    res = res.detach().numpy()
    res = transl_scaller.inverse_transform(res)

    return res[0]


def generate_transformation_from_shift(shift, name):
    numpy_array = np.eye(4)
    numpy_array[:3, 3] = -shift
    matrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            matrix.SetElement(i, j, numpy_array[i, j])

    transformNode = slicer.vtkMRMLLinearTransformNode()
    transformNode.SetName(name)
    transformNode.SetAndObserveMatrixTransformToParent(matrix)
    slicer.mrmlScene.AddNode(transformNode)


def slicer_transform_electrode(side):
    transform_name = "right_shift" if side == "right" else "left_shift"
    transform = slicer.util.getNode(transform_name)


def display_mesh(mesh, node_name):
    modelNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode')
    modelNode.SetAndObservePolyData(mesh)
    modelNode.SetDisplayVisibility(True)
    modelNode.SetName(node_name)

    displayNode = modelNode.GetDisplayNode()
    if displayNode is None:
        displayNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLModelDisplayNode")
        slicer.mrmlScene.AddNode(displayNode)
        modelNode.SetAndObserveDisplayNodeID(displayNode.GetID())
    displayNode.SetScalarVisibility(1)
    displayNode.SetVisibility3D(1)
    displayNode.SetVisibility2D(1)
    displayNode.SetOpacity(0.3)
    displayNode.Modified()
    return modelNode


def loadNiiImage(file_path):
    # Load an image and display it in Slicer
    image_node = slicer.util.loadVolume(file_path)
    slicer.util.setSliceViewerLayers(background=image_node)
    return image_node


#
# load_niftyLogic
#

class load_niftyLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return load_niftyParameterNode(super().getParameterNode())

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
        logging.info('Processing started')

        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        cliParams = {
            'InputVolume': inputVolume.GetID(),
            'OutputVolume': outputVolume.GetID(),
            'ThresholdValue': imageThreshold,
            'ThresholdType': 'Above' if invert else 'Below'
        }
        cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True,
                                 update_display=showResult)
        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime - startTime:.2f} seconds')


#
# load_niftyTest
#

class load_niftyTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_load_nifty1()

    def test_load_nifty1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
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
        inputVolume = SampleData.downloadSample('load_nifty1')
        self.delayDisplay('Loaded test data set')

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = load_niftyLogic()

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

        self.delayDisplay('Test passed')
