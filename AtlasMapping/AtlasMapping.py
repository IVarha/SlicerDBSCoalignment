import logging
import os
import subprocess
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
    if not os.path.exists(nifti_path):
        raise FileNotFoundError(f"Segmentation resource not found: {nifti_path}")
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



def convert_model_to_segmentation(model_node, reference_volume_node=None) -> vtkMRMLLabelMapVolumeNode:
    labelmap_volume_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    segmentation_node.CreateDefaultDisplayNodes()
    if reference_volume_node:
        segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(reference_volume_node)
        segmentation = segmentation_node.GetSegmentation()
        segmentation.SetConversionParameter("Oversampling factor", "1")
        segmentation.SetConversionParameter("Crop to reference image geometry", "1")
    slicer.modules.segmentations.logic().ImportModelToSegmentationNode(model_node, segmentation_node)
    if reference_volume_node:
        segmentation_node.CreateBinaryLabelmapRepresentation()
        slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(
            segmentation_node,
            labelmap_volume_node,
            reference_volume_node,
            slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY,
        )
    else:
        slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
            segmentation_node,
            labelmap_volume_node,
        )

    slicer.mrmlScene.RemoveNode(segmentation_node)
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


def create_image_only_bspline_parameters(input_params_path, output_params_path):
    with open(input_params_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    text = text.replace(
        '(Registration "MultiMetricMultiResolutionRegistration")',
        '(Registration "MultiResolutionRegistration")'
    )
    text = text.replace(
        '(Metric "AdvancedNormalizedCorrelation" "CorrespondingPointsEuclideanDistanceMetric" )',
        '(Metric "AdvancedNormalizedCorrelation")'
    )
    text = text.replace(
        '(ImageSampler "RandomCoordinate")',
        '(ImageSampler "RandomSparseMask")'
    )
    text = text.replace(
        '(MaximumStepLength 1.0 1.0 1.0 1.0 1.0 1.0 0.5)',
        '(MaximumStepLength 0.5 0.4 0.3 0.2 0.15 0.1 0.05)'
    )

    filtered_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("(Metric1Weight"):
            continue
        if stripped.startswith("(Metric2Weight"):
            continue
        filtered_lines.append(line)

    if not any(line.strip().startswith("(CheckNumberOfSamples") for line in filtered_lines):
        filtered_lines.append('(CheckNumberOfSamples "false")')

    with open(output_params_path, "w", encoding="utf-8") as f:
        f.write("\n".join(filtered_lines) + "\n")


def create_signed_distance_map_nifti(input_path, output_path):
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError("SimpleITK is required to create distance maps for mask registration.") from exc

    binary_image = sitk.ReadImage(input_path)
    distance_image = sitk.SignedMaurerDistanceMap(
        sitk.Cast(binary_image > 0, sitk.sitkUInt8),
        insideIsPositive=False,
        squaredDistance=False,
        useImageSpacing=True,
    )
    sitk.WriteImage(distance_image, output_path)


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

    def _ensureLogic(self, show_errors=True) -> bool:
        if self.logic:
            return True
        try:
            self.logic = AtlasMappingLogic()
            return True
        except Exception as exc:
            logging.exception("Failed to initialize AtlasMapping logic")
            self.logic = None
            if show_errors:
                slicer.util.errorDisplay(f"Failed to initialize AtlasMapping logic:\n{exc}")
            return False

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
        self._ensureLogic(show_errors=False)

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        if self.logic:
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
        if not self._ensureLogic(show_errors=False):
            return

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
        if (
            self._parameterNode
            and self._parameterNode.inputMesh
            and self._parameterNode.invertedVolume
        ):
            self.ui.applyButton.toolTip = _("Compute atlas mapping using selected input transform")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select input mesh and input transform node")
            self.ui.applyButton.enabled = False

    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        if not self._ensureLogic():
            return
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
            result_transform = self.logic.process(meshN, self.ui.invertedOutputSelector.currentNode())
            if self.logic.lastTargetIsLeft:
                self.logic._apply_x_mirror_and_harden(seg_1)
            if self.logic.lastPrealignmentTranslation is not None:
                self.logic._apply_translation_and_harden(seg_1, self.logic.lastPrealignmentTranslation)

            # set and observe transform node to segment
            seg_1.SetAndObserveTransformNodeID(result_transform.GetID())
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
        self.lastTargetIsLeft = False
        self.lastPrealignmentTranslation = None

    def getParameterNode(self):
        return AtlasMappingParameterNode(super().getParameterNode())

    @staticmethod
    def _mesh_centroid_x(model_node: vtkMRMLModelNode) -> float:
        points = model_node.GetPolyData().GetPoints()
        n = points.GetNumberOfPoints()
        if n == 0:
            return 0.0
        xs = np.empty(n, dtype=np.float64)
        for i in range(n):
            xs[i] = points.GetPoint(i)[0]
        return float(xs.mean())

    @staticmethod
    def _mask_moving_volume_to_hemisphere(moving_volume_node, keep_right_hemisphere: bool):
        image_array = slicer.util.arrayFromVolume(moving_volume_node)
        non_zero = np.argwhere(image_array > 0)
        if non_zero.size == 0:
            return

        ijk_to_ras_vtk = vtk.vtkMatrix4x4()
        moving_volume_node.GetIJKToRASMatrix(ijk_to_ras_vtk)
        ijk_to_ras = np.array(
            [[ijk_to_ras_vtk.GetElement(r, c) for c in range(4)] for r in range(4)],
            dtype=np.float64,
        )

        # array indices are [k, j, i]; convert to IJK for matrix multiplication
        ijk_h = np.c_[non_zero[:, 2], non_zero[:, 1], non_zero[:, 0], np.ones(non_zero.shape[0])]
        ras = ijk_h @ ijk_to_ras.T
        x_ras = ras[:, 0]

        if keep_right_hemisphere:
            drop = x_ras < 0.0
        else:
            drop = x_ras > 0.0

        to_drop = non_zero[drop]
        if to_drop.size > 0:
            image_array[to_drop[:, 0], to_drop[:, 1], to_drop[:, 2]] = 0
            slicer.util.arrayFromVolumeModified(moving_volume_node)

    @staticmethod
    def _non_zero_hemisphere_counts(volume_node):
        image_array = slicer.util.arrayFromVolume(volume_node)
        non_zero = np.argwhere(image_array > 0)
        if non_zero.size == 0:
            return 0, 0

        ijk_to_ras_vtk = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASMatrix(ijk_to_ras_vtk)
        ijk_to_ras = np.array(
            [[ijk_to_ras_vtk.GetElement(r, c) for c in range(4)] for r in range(4)],
            dtype=np.float64,
        )
        ijk_h = np.c_[non_zero[:, 2], non_zero[:, 1], non_zero[:, 0], np.ones(non_zero.shape[0])]
        ras = ijk_h @ ijk_to_ras.T
        x_ras = ras[:, 0]
        left_count = int(np.count_nonzero(x_ras < 0.0))
        right_count = int(np.count_nonzero(x_ras > 0.0))
        return left_count, right_count

    @staticmethod
    def _non_zero_voxel_count(volume_node) -> int:
        image_array = slicer.util.arrayFromVolume(volume_node)
        return int(np.count_nonzero(image_array))

    @staticmethod
    def _apply_x_mirror_and_harden(node):
        mirror_transform = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "AtlasMapping_MirrorX")
        matrix = vtk.vtkMatrix4x4()
        matrix.Identity()
        matrix.SetElement(0, 0, -1.0)
        mirror_transform.SetMatrixTransformToParent(matrix)
        node.SetAndObserveTransformNodeID(mirror_transform.GetID())
        slicer.vtkSlicerTransformLogic().hardenTransform(node)
        slicer.mrmlScene.RemoveNode(mirror_transform)

    def _prepare_moving_atlas(self, moving_volume_node, target_is_left: bool):
        left_count, right_count = self._non_zero_hemisphere_counts(moving_volume_node)

        # Unilateral atlas resources need mirroring for the opposite side.
        if target_is_left and left_count == 0 and right_count > 0:
            self._apply_x_mirror_and_harden(moving_volume_node)
            return
        if (not target_is_left) and right_count == 0 and left_count > 0:
            self._apply_x_mirror_and_harden(moving_volume_node)
            return

        # Bilateral atlases are reduced to the requested hemisphere.
        if left_count > 0 and right_count > 0:
            self._mask_moving_volume_to_hemisphere(moving_volume_node, not target_is_left)

    @staticmethod
    def _ensure_non_zero_spacing(volume_node):
        spacing = list(volume_node.GetSpacing())
        updated = False
        for i, s in enumerate(spacing):
            if s is None or s <= 0.0:
                spacing[i] = 1.0
                updated = True
        if updated:
            volume_node.SetSpacing(spacing)

    @staticmethod
    def _pad_degenerate_volume_axes(volume_node, min_dim=3):
        image_data = volume_node.GetImageData()
        if not image_data:
            return
        dim_i, dim_j, dim_k = image_data.GetDimensions()
        target_i = max(dim_i, min_dim)
        target_j = max(dim_j, min_dim)
        target_k = max(dim_k, min_dim)
        if (target_i, target_j, target_k) == (dim_i, dim_j, dim_k):
            return

        pad_i_before = (target_i - dim_i) // 2
        pad_i_after = target_i - dim_i - pad_i_before
        pad_j_before = (target_j - dim_j) // 2
        pad_j_after = target_j - dim_j - pad_j_before
        pad_k_before = (target_k - dim_k) // 2
        pad_k_after = target_k - dim_k - pad_k_before

        arr = slicer.util.arrayFromVolume(volume_node)
        padded = np.pad(
            arr,
            ((pad_k_before, pad_k_after), (pad_j_before, pad_j_after), (pad_i_before, pad_i_after)),
            mode="constant",
            constant_values=0,
        )
        slicer.util.updateVolumeFromArray(volume_node, padded)

        ijk_to_ras_vtk = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASMatrix(ijk_to_ras_vtk)
        ijk_to_ras = np.array(
            [[ijk_to_ras_vtk.GetElement(r, c) for c in range(4)] for r in range(4)],
            dtype=np.float64,
        )
        pad_vec = np.array([pad_i_before, pad_j_before, pad_k_before], dtype=np.float64)
        shift = ijk_to_ras[:3, :3] @ pad_vec
        ijk_to_ras[:3, 3] -= shift

        updated_m = vtk.vtkMatrix4x4()
        for r in range(4):
            for c in range(4):
                updated_m.SetElement(r, c, float(ijk_to_ras[r, c]))
        volume_node.SetIJKToRASMatrix(updated_m)

    @staticmethod
    def _pad_volume_to_include_bounds(volume_node, target_bounds, margin_voxels=8):
        if not volume_node or not volume_node.GetImageData() or not target_bounds:
            return

        volume_bounds = AtlasMappingLogic._node_bounds(volume_node)
        if not volume_bounds:
            return

        spacing = [abs(s) if s else 1.0 for s in volume_node.GetSpacing()]
        pad_i_before = int(np.ceil(max(0.0, volume_bounds[0] - target_bounds[0]) / spacing[0])) + (margin_voxels if target_bounds[0] < volume_bounds[0] else 0)
        pad_i_after = int(np.ceil(max(0.0, target_bounds[1] - volume_bounds[1]) / spacing[0])) + (margin_voxels if target_bounds[1] > volume_bounds[1] else 0)
        pad_j_before = int(np.ceil(max(0.0, volume_bounds[2] - target_bounds[2]) / spacing[1])) + (margin_voxels if target_bounds[2] < volume_bounds[2] else 0)
        pad_j_after = int(np.ceil(max(0.0, target_bounds[3] - volume_bounds[3]) / spacing[1])) + (margin_voxels if target_bounds[3] > volume_bounds[3] else 0)
        pad_k_before = int(np.ceil(max(0.0, volume_bounds[4] - target_bounds[4]) / spacing[2])) + (margin_voxels if target_bounds[4] < volume_bounds[4] else 0)
        pad_k_after = int(np.ceil(max(0.0, target_bounds[5] - volume_bounds[5]) / spacing[2])) + (margin_voxels if target_bounds[5] > volume_bounds[5] else 0)

        if not any((pad_i_before, pad_i_after, pad_j_before, pad_j_after, pad_k_before, pad_k_after)):
            return

        arr = slicer.util.arrayFromVolume(volume_node)
        padded = np.pad(
            arr,
            ((pad_k_before, pad_k_after), (pad_j_before, pad_j_after), (pad_i_before, pad_i_after)),
            mode="constant",
            constant_values=0,
        )
        slicer.util.updateVolumeFromArray(volume_node, padded)

        ijk_to_ras_vtk = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASMatrix(ijk_to_ras_vtk)
        ijk_to_ras = np.array(
            [[ijk_to_ras_vtk.GetElement(r, c) for c in range(4)] for r in range(4)],
            dtype=np.float64,
        )
        pad_vec = np.array([pad_i_before, pad_j_before, pad_k_before], dtype=np.float64)
        shift = ijk_to_ras[:3, :3] @ pad_vec
        ijk_to_ras[:3, 3] -= shift

        updated_m = vtk.vtkMatrix4x4()
        for r in range(4):
            for c in range(4):
                updated_m.SetElement(r, c, float(ijk_to_ras[r, c]))
        volume_node.SetIJKToRASMatrix(updated_m)

    @staticmethod
    def _bounds_center(bounds):
        if not bounds:
            return None
        return np.array(
            [
                0.5 * (bounds[0] + bounds[1]),
                0.5 * (bounds[2] + bounds[3]),
                0.5 * (bounds[4] + bounds[5]),
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _apply_translation_and_harden(node, translation_xyz):
        if translation_xyz is None:
            return
        translation_transform = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "AtlasMapping_Prealign")
        matrix = vtk.vtkMatrix4x4()
        matrix.Identity()
        matrix.SetElement(0, 3, float(translation_xyz[0]))
        matrix.SetElement(1, 3, float(translation_xyz[1]))
        matrix.SetElement(2, 3, float(translation_xyz[2]))
        translation_transform.SetMatrixTransformToParent(matrix)
        node.SetAndObserveTransformNodeID(translation_transform.GetID())
        slicer.vtkSlicerTransformLogic().hardenTransform(node)
        slicer.mrmlScene.RemoveNode(translation_transform)

    @staticmethod
    def _volume_supports_bspline(volume_node) -> bool:
        image_data = volume_node.GetImageData()
        if not image_data:
            return False
        dims = image_data.GetDimensions()
        # BSpline fitting becomes unstable/invalid if any axis is effectively flat.
        return all(d > 1 for d in dims)

    @staticmethod
    def _run_brainsfit(parameters):
        cli_node = slicer.cli.runSync(slicer.modules.brainsfit, None, parameters)
        status = cli_node.GetStatusString()
        if status not in ("Completed", "Completed with warnings"):
            error_text = cli_node.GetErrorText() if hasattr(cli_node, "GetErrorText") else ""
            raise RuntimeError(f"BRAINSFit failed with status '{status}'.\n{error_text}")
        if status == "Completed with warnings":
            error_text = cli_node.GetErrorText() if hasattr(cli_node, "GetErrorText") else ""
            if error_text and "ExceptionObject" in error_text:
                raise RuntimeError(f"BRAINSFit failed with status '{status}'.\n{error_text}")
        return cli_node

    @staticmethod
    def _tail_log(log_path, max_lines=40):
        if not os.path.exists(log_path):
            return None
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            return "".join(lines[-max_lines:]).strip()
        except Exception as exc:
            return f"<failed to read log: {exc}>"

    @staticmethod
    def _volume_geom(volume_node):
        if not volume_node or not volume_node.GetImageData():
            return "no-image-data"
        return f"dims={volume_node.GetImageData().GetDimensions()}, spacing={volume_node.GetSpacing()}"

    @staticmethod
    def _node_bounds(node):
        bounds = [0.0] * 6
        if hasattr(node, "GetRASBounds"):
            node.GetRASBounds(bounds)
            return tuple(bounds)
        return None

    @staticmethod
    def _transform_is_identity(transform_node, tolerance=1e-6) -> bool:
        if not transform_node or not transform_node.IsA("vtkMRMLLinearTransformNode"):
            return False
        matrix = vtk.vtkMatrix4x4()
        transform_node.GetMatrixTransformToParent(matrix)
        for r in range(4):
            for c in range(4):
                expected = 1.0 if r == c else 0.0
                if abs(matrix.GetElement(r, c) - expected) > tolerance:
                    return False
        return True

    @staticmethod
    def _bounds_overlap_score(bounds_a, bounds_b) -> float:
        if not bounds_a or not bounds_b:
            return 0.0
        score = 1.0
        for axis in range(3):
            a0, a1 = bounds_a[2 * axis], bounds_a[2 * axis + 1]
            b0, b1 = bounds_b[2 * axis], bounds_b[2 * axis + 1]
            overlap = min(a1, b1) - max(a0, b0)
            if overlap <= 0:
                return 0.0
            score *= overlap
        return float(score)

    def _make_registration_mesh_candidate(self, source_mesh, name, transform_chain):
        candidate = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
        poly_copy = vtk.vtkPolyData()
        poly_copy.DeepCopy(source_mesh.GetPolyData())
        candidate.SetAndObservePolyData(poly_copy)
        for transform_node in transform_chain:
            if transform_node:
                candidate.SetAndObserveTransformNodeID(transform_node.GetID())
                slicer.vtkSlicerTransformLogic().hardenTransform(candidate)
        return candidate

    @staticmethod
    def _linear_transform_matrix(transform_node):
        if not transform_node or not transform_node.IsA("vtkMRMLLinearTransformNode"):
            return None
        matrix = vtk.vtkMatrix4x4()
        transform_node.GetMatrixTransformToParent(matrix)
        return np.array([[matrix.GetElement(r, c) for c in range(4)] for r in range(4)], dtype=np.float64)

    @classmethod
    def _transforms_approximately_inverse(cls, transform_a, transform_b, tolerance=1e-4):
        matrix_a = cls._linear_transform_matrix(transform_a)
        matrix_b = cls._linear_transform_matrix(transform_b)
        if matrix_a is None or matrix_b is None:
            return False
        product = matrix_a @ matrix_b
        return np.allclose(product, np.eye(4), atol=tolerance)

    def process(self,
                inputMesh: vtkMRMLModelNode,
                inputBaseTransform: vtkMRMLTransformNode) -> vtkMRMLTransformNode:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputMesh: volume to be thresholded
        :param outputVolume: thresholding result
        """

        if not inputMesh or not inputBaseTransform:
            raise ValueError("Input or output volume is invalid")
        import time
        startTime = time.time()
        # generate temp folder
        import tempfile
        working_dir_obj = tempfile.TemporaryDirectory()
        working_dir = working_dir_obj.name
        print("temp dir", working_dir)
        fixed_volume_node = None
        moving_volume_node = None
        registration_mesh = None
        registration_mesh_candidates = []
        labelmap_segmentation = None
        result_transform = None
        target_is_left = False
        selected_candidate_name = None
        input_mesh_parent_transform = None
        elastix_params = []
        transformix_params = []
        try:
            self.lastPrealignmentTranslation = None
            required_resources = [
                resourcePath("eparams/STNlabel.nii.gz"),
            ]
            for resource_file in required_resources:
                if not os.path.exists(resource_file):
                    raise FileNotFoundError(f"Required resource not found: {resource_file}")

            if not inputMesh.GetPolyData() or not inputMesh.GetPolyData().GetPoints():
                raise ValueError("Input mesh has no polydata points.")
            if inputMesh.GetPolyData().GetPoints().GetNumberOfPoints() == 0:
                raise ValueError("Input mesh has zero points.")

            current_bounds = self._node_bounds(inputMesh)
            target_is_left = current_bounds[1] < 0.0 if current_bounds else self._mesh_centroid_x(inputMesh) < 0.0
            self.lastTargetIsLeft = target_is_left

            moving_volume_node = slicer.util.loadNodeFromFile(resourcePath("eparams/STNlabel.nii.gz"), "VolumeFile")
            if not moving_volume_node:
                raise RuntimeError("Failed to load temporary registration volumes for BRAINSFit.")

            moving_volume_node.SetName("AtlasMapping_MovingAtlasLabel")
            self._prepare_moving_atlas(moving_volume_node, target_is_left)
            if self._non_zero_voxel_count(moving_volume_node) == 0:
                raise RuntimeError("Prepared moving atlas volume is empty.")

            atlas_bounds = self._node_bounds(moving_volume_node)
            parent_transform = inputMesh.GetParentTransformNode()
            input_mesh_parent_transform = parent_transform
            candidate_specs = []
            # The displayed mesh may already be under a parent transform in Slicer even though
            # its polydata is not hardened. Registration has to consider that visible transform
            # chain, otherwise the rasterized fixed mask can be built in the wrong coordinate
            # system and drift away from the atlas before Elastix even starts.
            if parent_transform and inputBaseTransform and self._transforms_approximately_inverse(parent_transform, inputBaseTransform):
                candidate_specs.append(("parent_then_selected", [parent_transform, inputBaseTransform]))
            candidate_specs.append(("raw", []))
            if parent_transform:
                candidate_specs.append(("parent_only", [parent_transform]))
            if inputBaseTransform and not self._transform_is_identity(inputBaseTransform):
                candidate_specs.append(("selected_only", [inputBaseTransform]))
                if parent_transform and ("parent_then_selected", [parent_transform, inputBaseTransform]) not in candidate_specs:
                    candidate_specs.append(("parent_then_selected", [parent_transform, inputBaseTransform]))

            best_score = -1.0
            best_name = None
            for candidate_name, transform_chain in candidate_specs:
                candidate = self._make_registration_mesh_candidate(
                    inputMesh,
                    f"AtlasMapping_{candidate_name}",
                    transform_chain,
                )
                registration_mesh_candidates.append(candidate)
                score = self._bounds_overlap_score(self._node_bounds(candidate), atlas_bounds)
                if score > best_score:
                    best_score = score
                    best_name = candidate_name
                    registration_mesh = candidate

            if not registration_mesh:
                raise RuntimeError("Failed to construct registration mesh candidate.")
            selected_candidate_name = best_name
            logging.info(f"AtlasMapping selected registration candidate '{best_name}' with bounds score {best_score}.")

            registration_center = self._bounds_center(self._node_bounds(registration_mesh))
            atlas_center = self._bounds_center(self._node_bounds(moving_volume_node))
            if registration_center is not None and atlas_center is not None:
                self.lastPrealignmentTranslation = tuple((registration_center - atlas_center).tolist())
                self._apply_translation_and_harden(moving_volume_node, self.lastPrealignmentTranslation)

            # Expand the atlas reference grid before rasterizing the mesh. Without this,
            # Slicer can clip the mesh-derived labelmap to the original atlas FOV, which
            # then yields zero mask overlap and can cascade into native Elastix/Slicer crashes.
            self._pad_volume_to_include_bounds(moving_volume_node, self._node_bounds(registration_mesh))

            # Export mesh to labelmap using atlas geometry to avoid degenerate labelmap spacing.
            labelmap_segmentation = convert_model_to_segmentation(registration_mesh, moving_volume_node)
            fixed_volume_node = labelmap_segmentation
            if not fixed_volume_node:
                raise RuntimeError("Failed to create fixed registration labelmap from input mesh.")

            fixed_volume_node.SetName("AtlasMapping_FixedMeshLabel")
            self._ensure_non_zero_spacing(fixed_volume_node)
            self._ensure_non_zero_spacing(moving_volume_node)
            self._pad_degenerate_volume_axes(fixed_volume_node)
            self._pad_degenerate_volume_axes(moving_volume_node)
            fixed_mask_array = slicer.util.arrayFromVolume(fixed_volume_node) > 0
            moving_mask_array = slicer.util.arrayFromVolume(moving_volume_node) > 0
            fixed_voxels = int(np.count_nonzero(fixed_mask_array))
            overlap_voxels = int(np.count_nonzero(fixed_mask_array & moving_mask_array))
            if fixed_voxels == 0:
                raise RuntimeError(
                    "Prepared fixed registration labelmap is empty.\n"
                    f"Registration mesh bounds: {self._node_bounds(registration_mesh)}\n"
                    f"Moving atlas bounds: {self._node_bounds(moving_volume_node)}\n"
                    f"Fixed labelmap bounds: {self._node_bounds(fixed_volume_node)}"
                )
            if overlap_voxels == 0:
                logging.warning(
                    "Prepared fixed registration labelmap has zero voxel overlap with the moving atlas mask. "
                    "Proceeding with distance-map registration without binary masks. "
                    f"Selected candidate: {selected_candidate_name}; "
                    f"Registration mesh bounds: {self._node_bounds(registration_mesh)}; "
                    f"Moving atlas bounds: {self._node_bounds(moving_volume_node)}; "
                    f"Fixed labelmap bounds: {self._node_bounds(fixed_volume_node)}; "
                    f"Fixed voxels: {fixed_voxels}; "
                    f"Moving voxels: {int(np.count_nonzero(moving_mask_array))}; "
                    f"Overlap voxels: {overlap_voxels}"
                )

            fixed_volume_path = os.path.join(working_dir, "mesh_label.nii.gz")
            moving_volume_path = os.path.join(working_dir, "moving_atlas_label.nii.gz")
            if not slicer.util.saveNode(fixed_volume_node, fixed_volume_path):
                raise RuntimeError(f"Failed to save fixed registration volume: {fixed_volume_path}")
            if not slicer.util.saveNode(moving_volume_node, moving_volume_path):
                raise RuntimeError(f"Failed to save moving atlas volume: {moving_volume_path}")

            fixed_distance_path = os.path.join(working_dir, "mesh_distance.nii.gz")
            moving_distance_path = os.path.join(working_dir, "moving_atlas_distance.nii.gz")
            create_signed_distance_map_nifti(fixed_volume_path, fixed_distance_path)
            create_signed_distance_map_nifti(moving_volume_path, moving_distance_path)

            result_transform = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLTransformNode",
                f"{inputBaseTransform.GetName()}_AtlasMapping",
            )
            if selected_candidate_name == "raw" and input_mesh_parent_transform:
                result_transform.SetAndObserveTransformNodeID(input_mesh_parent_transform.GetID())

            try:
                import Elastix
            except ImportError as exc:
                raise RuntimeError(
                    "SlicerElastix is not available in this Slicer session. Restart Slicer after installing the extension."
                ) from exc

            elastix_logic = Elastix.ElastixLogic()
            elastix_module_dir = os.path.dirname(os.path.abspath(Elastix.__file__))
            elastix_param_dir = os.path.join(elastix_module_dir, "Resources", "RegistrationParameters")
            rigid_params_path = os.path.join(elastix_param_dir, "Parameters_RigidAMS.txt")
            bspline_params_path = os.path.join(elastix_param_dir, "Parameters_BSpline.txt")
            for params_path in (rigid_params_path, bspline_params_path):
                if not os.path.exists(params_path):
                    raise RuntimeError(f"Required Elastix parameter file not found: {params_path}")
            # Keep the live module on the exact SlicerElastix parameter pair that already
            # completed successfully on the exported distance maps. The earlier custom
            # parameter files were the source of repeated metric failures and native crashes.
            elastix_params = [
                "-f", fixed_distance_path,
                "-m", moving_distance_path,
                "-p", rigid_params_path,
                "-p", bspline_params_path,
                "-out", working_dir,
            ]
            elastix_process = elastix_logic.startElastix(elastix_params)
            elastix_logic.logProcessOutput(elastix_process)

            transformix_params = [
                "-in", moving_volume_path,
                "-tp", os.path.join(working_dir, "TransformParameters.1.txt"),
                "-def", "all",
                "-out", working_dir,
            ]
            transformix_process = elastix_logic.startTransformix(transformix_params)
            elastix_logic.logProcessOutput(transformix_process)

            output_transform_candidates = [
                os.path.join(working_dir, "deformationField.mhd"),
                os.path.join(working_dir, "deformationField.nii.gz"),
            ]
            output_transform_path = next((path for path in output_transform_candidates if os.path.exists(path)), None)
            if output_transform_path is None:
                raise RuntimeError(
                    "Transformix did not produce a deformation field. Checked: "
                    + ", ".join(output_transform_candidates)
                )
            elastix_logic.loadTransformFromFile(output_transform_path, result_transform)
        except subprocess.CalledProcessError as e:
            elastix_log_path = os.path.join(working_dir, "elastix.log")
            transformix_log_path = os.path.join(working_dir, "transformix.log")
            elastix_log_tail = self._tail_log(elastix_log_path)
            transformix_log_tail = self._tail_log(transformix_log_path)
            log_sections = []
            if elastix_log_tail is not None:
                log_sections.append(f"{elastix_log_path}\n{elastix_log_tail}")
            if transformix_log_tail is not None:
                log_sections.append(f"{transformix_log_path}\n{transformix_log_tail}")
            logs_text = "\n\n".join(log_sections) if log_sections else "<no elastix/transformix logs found>"
            raise RuntimeError(
                f"Elastix/Transformix failed with exit code {e.returncode}.\n"
                f"Fixed geometry: {self._volume_geom(fixed_volume_node)}\n"
                f"Moving geometry: {self._volume_geom(moving_volume_node)}\n"
                f"Elastix command args: {' '.join(elastix_params)}\n"
                f"Transformix command args: {' '.join(transformix_params)}\n"
                f"Temporary output directory: {working_dir}\n\n"
                f"Log tail:\n{logs_text}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Atlas mapping failed in temporary directory {working_dir}: {e}") from e
        finally:
            if fixed_volume_node:
                slicer.mrmlScene.RemoveNode(fixed_volume_node)
            if moving_volume_node:
                slicer.mrmlScene.RemoveNode(moving_volume_node)
            # fixed_volume_node is labelmap_segmentation; avoid double-remove
            if labelmap_segmentation and labelmap_segmentation != fixed_volume_node:
                slicer.mrmlScene.RemoveNode(labelmap_segmentation)
            for candidate in registration_mesh_candidates:
                if candidate:
                    slicer.mrmlScene.RemoveNode(candidate)
            # Keep temporary output available for troubleshooting.
            # Set cleanup() here once diagnostics are no longer needed.
            pass


        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")
        return result_transform


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
