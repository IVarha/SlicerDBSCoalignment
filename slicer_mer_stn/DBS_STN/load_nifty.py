import logging
import os
import pickle
from typing import Annotated, Optional

import numpy as np
import qt
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
from dbs_image_utils.nets import CenterDetector, CenterAndPCANet


def get_flirt_transformation_matrix(mat_file, src_file, dest_file, from_, to):
    im_src = fim.Image(src_file, loadData=False)
    im_dest = fim.Image(dest_file, loadData=False)
    forward_transf_fsl = fl.readFlirt(mat_file)
    return fl.fromFlirt(forward_transf_fsl, im_src, im_dest, from_, to)


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
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
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
        self.ui.wmSegmentationButton.clicked.connect(self.onApplyWMSeg)
        self.ui.wmIntensityNormButton.clicked.connect(self.onApplyIntensity)
        self.ui.twoStepCoregistrationButton.clicked.connect(self.onTwoStepCoregistration)

        self.ui.inputPathSelector.connect('currentPathChanged(QString)', self.onInputFolderSelect)
        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

        # create temp workdir
        self._create_temp_folder()

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
        ### dont reload if image exist in scene
        # all_images = slicer.mrmlScene.GetNodesByClass("vtkMRMLScalarVolumeNode")

        try:
            node = slicer.util.getNode(key)
            slicer.mrmlScene.RemoveNode(node)
            # node is found
        except:  # node not found
            pass  # todo change this method to get t2
        node = loadNiiImage(str(self.processing_folder / newvalue))
        node.SetName(key)
        # all_images_names = [node.GetName() for node in all_images]
        # if not newvalue.split(".")[0] in all_images_names:
        #     loadNiiImage(str(self.processing_folder / newvalue))
        # # end

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
        slicer_preprocessing.wm_segmentation(t1=str(self.processing_folder / self.t1),
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
        # node = slicer.util.getNode("TransformParameters.0")
        transfortm_node.SetName("to_mni")
        t2_image = slicer.util.getNode("t2_normalised")
        t2_image.SetAndObserveTransformNodeID(transfortm_node.GetID())
        t2_image.HardenTransform()
        self.onSegmentationButtonClicked()

        # node.SetName("to_mni")

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

    def segment_side(self, center_pred_point,image_coords, mirror=False):
        # load mesh
        mesh = vtk.vtkOBJReader()
        mesh.SetFileName(self.resourcePath('nets/3.obj'))
        mesh.Update()
        mesh = mesh.GetOutput()

        mm_offset = 2
        t2 = slicer.util.getNode("t2_normalised")
        image_processor = SlicerImage(t2.GetImageData())

        transform_ras_to_ijk = vtk.vtkMatrix4x4()
        t2.GetIJKToRASMatrix(transform_ras_to_ijk)
        transform_ras_to_ijk.Invert()

        shape_im = image_processor.compute_image_at_pts(points=image_coords,
                                                        transform_ras_to_ijk=transform_ras_to_ijk)
        shape_im = np.reshape(shape_im, (self.shape_label_mask.n_x,
                                         self.shape_label_mask.n_y, self.shape_label_mask.n_z))
        shape_im = self.shape_histogram.apply_normalization(shape_im)
        shape_im = torch.from_numpy(np.expand_dims(np.expand_dims(shape_im, axis=0), axis=0)).type(torch.float32)

        print("segm start")
        out_pcas = self.shape_predictor(shape_im, False)
        print("segm finished")
        off_cent = (out_pcas[:, -3:] * 2 * mm_offset) - mm_offset  # reshape center offset
        # compute resulting center
        result_center = center_pred_point + off_cent.detach().numpy()
        pcas = out_pcas[:, :-3]
        shape = self.shape_pca_res[0].inverse_transform(pcas.detach().numpy())[0]
        shape = np.reshape(shape, (int(shape.shape[0] / 3), 3)) + result_center
        print(shape.shape)
        if mirror:
            shape = shape * [-1, 1, 1]
            result_center = result_center * [-1, 1, 1]

        return change_mesh(mesh, shape), result_center

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

            mesh_orig, pts_orig = self.segment_side(res_a1,image_coords_orig)
            mesh_mirr, pts_mirr = self.segment_side(res_a0, image_coords_mirr,True)
            print(pts_orig,pts_mirr)
            pts_mirr = pts_mirr[0]
            pts_orig = pts_orig[0]
            self.center_orig = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode')
            self.center_orig.AddControlPointWorld(pts_mirr[0], pts_mirr[1], pts_mirr[2])
            self.center_orig.AddControlPointWorld(pts_orig[0], pts_orig[1], pts_orig[2])


            self.mesh1 = display_mesh(mesh_orig,"Mesh left")
            self.mesh2 = display_mesh(mesh_mirr, "Mesh right")



        except Exception as e:
            raise e
            # print(5)

def display_mesh(mesh, node_name):
    modelNode  = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode')
    modelNode .SetAndObservePolyData(mesh)
    modelNode .SetDisplayVisibility(True)
    modelNode .SetName(node_name)
    displayNode = modelNode.GetDisplayNode()
    if displayNode is None:
        displayNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLModelDisplayNode")
        slicer.mrmlScene.AddNode(displayNode)
        modelNode.SetAndObserveDisplayNodeID(displayNode.GetID())
    displayNode.SetScalarVisibility(1)
    displayNode.SetVisibility3D(1)
    displayNode.SetVisibility2D(1)
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
