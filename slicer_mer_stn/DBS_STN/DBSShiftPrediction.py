import logging

import pandas as pd
import slicer

import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated, Optional, Tuple, List

from torch import nn
from torch.nn import functional as F

try:
    import fsl.data.image as fim
except ImportError:
    slicer.util.pip_install('fslpy')
    import fsl.data.image as fim
import fsl.transform.flirt as fl

try:
    import mer_lib.artefact_detection as ad
except ImportError:
    slicer.util.pip_install(r'C:\\Users\\h492884\\PycharmProjects\\MER_lib')
    import mer_lib.artefact_detection as ad

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

from MRMLCorePython import vtkMRMLVolumeArchetypeStorageNode, vtkMRMLModelNode, vtkMRMLTransformNode, vtkMRMLTextNode, \
    vtkMRMLSubjectHierarchyNode, vtkMRMLLinearTransformNode

try:
    from dbs_image_utils.mask import SubcorticalMask
except ImportError:
    slicer.util.pip_install(r'C:\\Users\\h492884\\PycharmProjects\\dbs_pure_lib')
    from dbs_image_utils.mask import SubcorticalMask
from dbs_image_utils.nets import CenterDetector, CenterAndPCANet, TransformerClassifier
from mer_lib.data import MER_data
from sklearn.preprocessing import MinMaxScaler
from slicer import vtkMRMLScalarVolumeNode
from slicer.ScriptedLoadableModule import *
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)
from slicer.util import VTKObservationMixin

from Lib import slicer_preprocessing
from Lib.image_utils import SlicerImage
from Lib.mer_support import EntryTarget, Point, cross_generation_mni, ElectrodeRecord, \
    clasify_mers, extract_points_from_mesh
from Lib.utils_file import get_images_in_folder
from Lib.visualisiation import LeadORLogic


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
        data.extracted_features = (extracted_features - extracted_features.min()) / (
                extracted_features.max() - extracted_features.min())

        return data

    runner = proc.Processor()
    runner.set_processes([ad.covariance_method, fe.nrms_calculation, nrms_normalisation, min_max_scaler])
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


def check_points_inside_vtk_mesh(mesh, points):
    """
    Check if points are inside a VTK mesh.

    Parameters:
    mesh (vtk.vtkPolyData): The VTK mesh.
    points (np.ndarray): An array of points.

    Returns:
    np.ndarray: An array of booleans indicating whether each point is inside the mesh.
    """

    # Create a vtkPoints object from the numpy array
    vtk_points = vtk.vtkPoints()
    for point in points:
        vtk_points.InsertNextPoint(point)

    # Create a vtkPolyData object from the vtkPoints
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)

    # Use vtkSelectEnclosedPoints to check if the points are inside the mesh
    select_enclosed_points = vtk.vtkSelectEnclosedPoints()
    select_enclosed_points.SetInputData(polydata)
    select_enclosed_points.SetSurfaceData(mesh)
    select_enclosed_points.Update()

    # Get the output of vtkSelectEnclosedPoints as a numpy array
    is_inside = np.zeros(points.shape[0], dtype=bool)
    for i in range(points.shape[0]):
        is_inside[i] = select_enclosed_points.IsInside(i)

    return is_inside


def generate_correctly_placed_bitmap(labeled_points, current_output) -> torch.Tensor:
    """
    Generates a correctly placed bitmap based on the labeled points and current output.

    Args:
        labeled_points (numpy.ndarray): Array of labeled points.
        current_output (numpy.ndarray): Array of current output.

    Returns:
        torch.Tensor: Tensor representing the correctly placed bitmap.
    """
    x = []
    # print(labeled_points.shape)
    # print(current_output.shape)
    for i in range(len(current_output)):
        if current_output[i] != labeled_points[i]:  # not equal
            if labeled_points[i] == 1: # if labeled as inside the mesh but put to outside
                x.append(2)
            else: # if labeled as outside the mesh but
                x.append(1)
        else:
            x.append(0)

    return torch.from_numpy(np.array(x))


def distances_to_mesh(points, mesh_vertices):
    """
    Calculate the distances from given points to a mesh composed of vertices.

    Args:
        points (torch.Tensor): Tensor of shape (N, 3) representing N points.
        mesh_vertices (torch.Tensor): Tensor of shape (M, 3) representing M vertices of the mesh.

    Returns:
        torch.Tensor: Tensor of shape (N,) containing the distances from each point to the mesh.
    """
    # Expand dimensions of points and mesh vertices for broadcasting
    points_expanded = points.unsqueeze(1)  # Shape: (N, 1, 3)
    mesh_vertices_expanded = mesh_vertices.unsqueeze(0)  # Shape: (1, M, 3)

    # Calculate distances between points and mesh vertices
    distances = torch.norm(points_expanded - mesh_vertices_expanded, dim=-1)  # Shape: (N, M)

    # Compute the minimum distance for each point
    min_distances, _ = distances.min(dim=-1)  # Shape: (N,)

    return min_distances


def optimisation_criterion(orig_points, in_out, shift, scalling, mesh: vtk.vtkPolyData, verbose=False):
    """
    Calculate the criterion value for a given set of original points, in_out values, shift vector, and mesh.
    Args:
        orig_points (torch.Tensor): The original points.
        in_out (torch.Tensor): The in_out values.
        shift (torch.Tensor): The shift vector. TARGET
        mesh (cMesh): The mesh object.

    Returns:
        torch.Tensor: The criterion value.
    """
    if not isinstance(shift, torch.Tensor):
        shift = torch.tensor(shift, dtype=torch.float32)
        scalling = torch.tensor(scalling, dtype=torch.float32)
    orig_points = orig_points[:, :3]
    mean_op = orig_points.mean(dim=0)
    if isinstance(orig_points, np.ndarray):
        orig_points = torch.from_numpy(orig_points)
    # print(orig_points)
    new_points = (orig_points - mean_op)*scalling - shift + mean_op

    points_inside_posttrans = check_points_inside_vtk_mesh(mesh, new_points.detach().numpy())

    weight = generate_correctly_placed_bitmap(in_out, points_inside_posttrans)

    if verbose:
        print("Wrongly marked points: ", str((weight > 0).sum()))
        # print(f'    {(weight>0).sum():.2f}')

    mp = extract_points_from_mesh(mesh)
    # print ("mesh pt 0", mp[:3])
    mesh_pts = np.reshape(mp, (len(mp) // 3, 3))

    mesh_pts = torch.from_numpy(mesh_pts)

    # print(torch.abs(distances_to_mesh(new_points,mesh_pts)))

    result_error = (weight * torch.abs(distances_to_mesh(new_points, mesh_pts))).sum()

    result_error = result_error/ weight.shape[0]# / (weight > 0).sum()

    result_error = result_error #+ weight.sum()

    return result_error.item()


def subgradient_optimisation_criterion(shift,scaling, function):
    loss = function(shift,scaling)
    loss.backward()
    subgradient_shift = shift.grad.sign()
    subgradient_scaling = scaling.grad.sign()
    shift.grad.zero_()  # Clear gradients for the next iteration
    scaling.grad.zero_()  # Clear gradients for the next iteration
    return subgradient_shift, subgradient_scaling


def optimise_mer_signal(mer_data: torch.Tensor,
                        origin_shift: torch.Tensor,
                        mesh, classes: torch.Tensor,
                        lambda1=1.0,
                        distance=0.2,
                        learning_rate=0.004):
    """
    optimise the signal of the mer data to get the best overlap with the mesh
    input: mer_data: torch tensor unscalled mer data
    classes: classified signals (same length as mer_data)

    """
    # convert mer_data to torch tensor

    optimise_f = lambda x, y, verbose=False: (
             lambda1* optimisation_criterion(mer_data, classes, shift=x, scalling=y, mesh=mesh, verbose=verbose)
            + np.linalg.norm(x)
            + distance*(np.linalg.norm(y - 1))
            #+ (torch.linalg.norm(x) if torch.linalg.norm(x) > distance else 1 / torch.linalg.norm(x))
    ).item()

    x = origin_shift.clone().detach().requires_grad_(True)
    y = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)
    print("------------------------------")
    print("Origin estimated value of shift:", x)




    #

    # Define the optimizer

    min_value = 0.7
    max_value = 1.3

    op_ = lambda x,v=False: optimise_f(x[:3], x[3:], v)
    x = origin_shift.clone().detach().numpy().reshape((3)).tolist() + [1.0, 1.0, 1.0]
    x = np.array(x)
    print("Origin estimated value of the function:", op_(x,True))
    print("Origin estimated distance", np.linalg.norm(x[:3]))
    from scipy import optimize as opt

    bounds = [(-3, 3), (-3, 3), (-3, 3),
              (min_value, max_value), (min_value, max_value), (min_value, max_value)]

    res = opt.differential_evolution(op_, bounds, disp=True,x0=x,polish=False,seed=42)
    # res = opt.minimize(op_, res.x,method='Powell',options={'disp': True},
    #                  bounds=bounds)

    x = res.x
    print(res)
    print("\n\n")
    print("Optimized value of x:", x, " y:", y)
    print("final estimated distance", np.linalg.norm(x[:3]))
    print("Optimized value of the function:", op_(torch.from_numpy(x),True))
    print("------------------------------")

    return x[:3], torch.tensor([1.0, 1.0, 1.0], requires_grad=True).detach().numpy()
    # # Optimization loop
    # for i in range(2000):
    #     subgradient = subgradient_optimisation_criterion(x,y, op_)
    #     x.data -= learning_rate * subgradient[0]
    #     y.data -= learning_rate * subgradient[1]
    #     # clip the values
    #     y.data = torch.clamp(y.data, min_value, max_value)
    #     if i>0 and i % 100 == 0:
    #         learning_rate = learning_rate * 0.9
    #         # Print intermediate results every 100 iterations
    #         print(f'Iteration {i}: {op_(x,y)}')
    #
    # print("\n\n")
    # print("Optimized value of x:", x, " y:", y)
    # print("final estimated distance", torch.linalg.norm(x))
    # print("Optimized value of the function:", optimise_f(x, y, True).item())
    # print("------------------------------")
    # return x.detach().numpy(),torch.tensor([1.0, 1.0, 1.0], requires_grad=True).detach().numpy() #y.detach().numpy()


#
# DBSShiftPrediction
#

class DBSShiftPrediction(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "DBS Subcortical electrode localisation"  # TODO: make this more human readable by adding spaces
        self.parent.categories = [
            "DBS"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = [
            "John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#DBSShiftPrediction">module documentation</a>.
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

    # DBSShiftPrediction1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='DBSShiftPrediction',
        sampleName='DBSShiftPrediction1',
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, 'DBSShiftPrediction1.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames='DBSShiftPrediction1.nrrd',
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums='SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
        # This node name will be used when the data set is loaded
        nodeNames='DBSShiftPrediction1'
    )

    # DBSShiftPrediction2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='DBSShiftPrediction',
        sampleName='DBSShiftPrediction2',
        thumbnailFileName=os.path.join(iconsPath, 'DBSShiftPrediction2.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames='DBSShiftPrediction2.nrrd',
        checksums='SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
        # This node name will be used when the data set is loaded
        nodeNames='DBSShiftPrediction2'
    )


#
# DBSShiftPredictionParameterNode
#

@parameterNodeWrapper
class DBSShiftPredictionParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """
    inputVolume: vtkMRMLLinearTransformNode
    inputFeatureText: vtkMRMLTextNode
    inputMesh: vtkMRMLModelNode



#
# DBSShiftPredictionWidget
#

def change_mesh(mesh, ch_pts):
    polys = mesh.GetPolys()
    pts = mesh.GetPoints()
    for i in range(pts.GetNumberOfPoints()):
        pts.SetPoint(i, ch_pts[i, 0], ch_pts[i, 1], ch_pts[i, 2])

    mesh.SetPoints(pts)
    return mesh


class DBSShiftPredictionWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
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


def compute_image_at_pts(image_processor, image_coords, transform_ras_to_ijk, result_shape):
    shape_im = image_processor.compute_image_at_pts(points=image_coords,
                                                    transform_ras_to_ijk=transform_ras_to_ijk)
    shape_im = np.reshape(shape_im, result_shape)
    return shape_im


def _read_pickle(filename):
    f = open(filename, 'rb')
    res = pickle.load(f)
    f.close()
    return res


def compute_shape(pca_transform, out_pcas, result_center):
    pcas = out_pcas[:, :-3]
    shape = pca_transform.inverse_transform(pcas.detach().numpy())[0]
    shape = np.reshape(shape, (int(shape.shape[0] / 3), 3)) + result_center
    return shape


class TransformerShiftPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, mesh_dim: int):
        super(TransformerShiftPredictor, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        # self.transformer = nn.GRU(input_dim, hidden_dim, batch_first=True) # GRU
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads, batch_first=True),
            num_layers
        )

        self.mesh_embedding_dim = nn.Linear(mesh_dim, 64)
        self.mesh_embedding_dim2 = nn.Linear(64, 64)

        self.out_transformer = nn.Linear(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim + 64, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 3)

    def forward(self, signals, mesh_pcas):
        x = signals
        # _, h_n = self.transformer(x) # GRU
        # out = self.out_transformer(h_n) # GRU
        x = torch.tanh(self.embedding(x))
        vals = self.transformer(x)
        out = self.out_transformer(torch.relu(vals))
        out = torch.mean(vals, axis=0)

        # print(vals.shape,out.shape)
        out = out.unsqueeze(0)
        embedded_mesh = F.relu(self.mesh_embedding_dim(mesh_pcas))
        embedded_mesh = F.relu(self.mesh_embedding_dim2(embedded_mesh))

        fused_features = torch.cat((out, embedded_mesh), dim=1)

        x = torch.relu(self.fc1(fused_features))
        x = torch.sigmoid(self.fc2(x))

        return x


def check_side(bds):
    if bds[0] < 0:
        return True
    else:
        return False


def _remove_previous_node(node_name):
    if slicer.mrmlScene.GetFirstNodeByName(node_name):
        slicer.mrmlScene.RemoveNode(slicer.mrmlScene.GetFirstNodeByName(node_name))


def compute_text(classes, distance_to_target, columns):
    # extract the classes
    classes = classes.detach().numpy()

    # create dict from columns as keys
    dict_cols = {col: [] for col in columns}
    i = 0
    d = len(distance_to_target)
    for col in columns:
        for j in range(d):
            dict_cols[col].append(int(classes[i * d + j][0]))
        i += 1
    res = pd.DataFrame(dict_cols)
    res['RecordingSiteDTT'] = distance_to_target
    # remove text node if exists

    # create text node
    text = vtkMRMLTextNode()
    text.SetText(res.to_csv(index=False))

    return text


class DBSShiftPredictionWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
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
        self.logic: MRI_MERLogic = None
        self._parameterNode : DBSShiftPredictionParameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/DBSShiftPrediction.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic: MRI_MERLogic = MRI_MERLogic()
        print("MER logic created")
        # self._create_temp_folder()
        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        # self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)
        self.ui.stn_inputSelector.currentNodeChanged.connect(self.onInputMeshSelected)
        self.ui.textinputSelector.currentNodeChanged.connect( self.onInputTextSelected)
        self.ui.transform_inputSelector.currentNodeChanged.connect(self.onInputTransformSelected)
        self.ui.merFinalButton.clicked.connect(self.on_calculate_shift)
        # self.ui.merApplyTransformation.clicked.connect(self.apply_transformation_mer)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

        # create temp workdir

        # self.lead_or_bind()
        # initialise variables

        self.mer_transforms = _read_pickle(self.resourcePath('nets/mer_pca_mesh.pkl'))

    def onInputTransformSelected(self, node: vtkMRMLTransformNode):
        self.to_mni = node

    def onInputTextSelected(self, node: vtkMRMLTextNode):
        self.text = node
        print('Text selected')
        #
        ### add event when the text is changed
        # self.text.AddObserver(vtk.vtkCommand.ModifiedEvent, self.onTextModified)

    def onTextModified(self, node):

        _remove_previous_node("LeadOR: Classes")
        # compute shift transformation
        res_transform, text_node = self.logic.predict_shift(node, self.side, self.to_mni, self.mesh_copy, self.pcas,
                                                            lambda1=100, distance=20, learning_rate=0.01)
        # create text node
        text_node.SetName("LeadOR: Classes")
        slicer.mrmlScene.AddNode(res_transform)
        slicer.mrmlScene.AddNode(text_node)

    def onInputMeshSelected(self, node: vtkMRMLModelNode):
        self.mesh1 = node
        # get cells bounds from node
        bds = [0] * 6
        node.GetBounds(bds)
        # check if the mesh is on the right or left side
        self.side = check_side(bds)

        # get pcas from the mesh
        self.pcas, mesh_points = self.logic.get_pcas_from_mesh(node, self.side)

        # create mesh copy
        mesh_copy = self.logic.create_mesh_copy(node)
        # change points of the mesh
        if self.side:
            mesh_points = np.reshape(mesh_points, (-1, 3))
            print(mesh_points.shape)
            mesh_copy = change_mesh(mesh_copy, mesh_points)
        self.mesh_copy = mesh_copy
        print(self.pcas)

    def cleanup(self) -> None:
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()
        # self._remove_temporary_folder()

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
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[DBSShiftPredictionParameterNode]) -> None:
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
        pass


    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.inputFeatureText and self._parameterNode.inputMesh:
            self.ui.merFinalButton.toolTip = "Compute output volume"
            self.ui.merFinalButton.enabled = True
        else:
            self.ui.merFinalButton.toolTip = "Select input and output volume nodes"
            self.ui.merFinalButton.enabled = False

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

    def _test_copytransform(self):
        sh = slicer.mrmlScene.GetSubjectHierarchyNode()
        sceneId = sh.GetSceneItemID()
        fold1 = sh.CreateFolderItem(sceneId, "folder1")
        # create a transform node
        transformNode = slicer.vtkMRMLLinearTransformNode()
        slicer.mrmlScene.AddNode(transformNode)
        transformNode.SetName("transform1")
        sh.SetItemParent(sh.GetItemByDataNode(transformNode), fold1)
        # create 2nd folder
        fold2 = sh.CreateFolderItem(sceneId, "folder2")

        # copy the transform node
        cp_transf = slicer.mrmlScene.CopyNode(transformNode)

        dtt_id = sh.GetItemByDataNode(cp_transf)
        sh.SetItemParent(dtt_id, fold2)

    def _copy_leadORIGTL_folder(self):
        """
        Copy the LeadORIGTL folder to the temporary working directory.
        """
        sh = slicer.mrmlScene.GetSubjectHierarchyNode()

        transformNode = vtkMRMLLinearTransformNode()
        slicer.mrmlScene.AddNode(transformNode)
        transformNode.Copy(slicer.util.getNode("LeadOR:DTT"))
        transformNode.SetName("DBS DTT")

        # Get Parent Transform Node
        par_transf = slicer.util.getNode("LeadOR:DTT").GetParentTransformNode()
        transformNode.SetAndObserveTransformNodeID(par_transf.GetID())

        # CREATE FOLDER FOR THE NEW TRANSFORM
        sceneId = sh.GetSceneItemID()
        fold1 = sh.CreateFolderItem(sceneId, "DBS-Transforms")
        dtt_id = sh.GetItemByDataNode(transformNode)
        sh.SetItemParent(dtt_id, fold1)

    def _remove_all_previous_copies(self):
        sh = slicer.mrmlScene.GetSubjectHierarchyNode()
        sceneId = sh.GetSceneItemID()
        child_tmp = []
        sh.GetItemChildren(sceneId, child_tmp)
        for id in child_tmp:

            if sh.GetItemLevel(id) == 'Folder':
                # print("Folder", sh.GetItemName(id))
                name = sh.GetItemName(id)
                if name.startswith("DBS-T") or name.startswith("DBS T"):
                    sh.RemoveItem(id)

    def copy_leador_electrodes(self):

        sh: vtkMRMLSubjectHierarchyNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        sceneId = sh.GetSceneItemID()

        # get all children of sc
        child_tmp = []
        sh.GetItemChildren(sceneId, child_tmp)
        ch2 = []
        sh.GetItemChildren(child_tmp[0], ch2)
        print(child_tmp)
        for id in (child_tmp + ch2):
            # check if is folder
            if sh.GetItemLevel(id) == 'Folder':
                print(sh.GetItemName(id))
                name = sh.GetItemName(id)
                if name.startswith("LeadOR T"):
                    print("Creating a folder for", name)
                    result_name = name.split("LeadOR ")[-1]

                    # create a copy of a folder and add it to the scene
                    new_id = sh.CreateFolderItem(sceneId, "DBS " + result_name)

                    # copy all children of origin folder to the new folder
                    child_tmp2 = []
                    sh.GetItemChildren(id, child_tmp2)
                    # copy all children

                    transform_id = None
                    transformed_ids = []
                    for id2 in child_tmp2:

                        id2_name = sh.GetItemName(id2)
                        id2_node = slicer.util.getNode(id2_name)

                        clonedItemID = slicer.modules.subjecthierarchy.logic().CloneSubjectHierarchyItem(sh,
                                                                                                         id2)
                        clonedNode = sh.GetItemDataNode(clonedItemID)

                        sh.SetItemParent(clonedItemID, new_id)
                        clonedNode.SetName(id2_name)
                        if clonedNode.IsA('vtkMRMLLinearTransformNode'):
                            transform_id = clonedItemID
                            print(111)
                        if id2_node.IsA('vtkMRMLTransformableNode'):
                            if id2_name.endswith("Tube Model"):
                                transf_plan = slicer.util.getNode("LeadOR:DTT").GetParentTransformNode()
                                clonedNode.SetAndObserveTransformNodeID(transf_plan.GetID())
                            else:
                                print(id2)
                                transform_node_tmp = id2_node.GetParentTransformNode()
                                # If a transform node is found, add it to the list
                                if transform_node_tmp:
                                    transformed_ids.append(clonedItemID)

                    # apply transform to copied nodes
                    transform_node = sh.GetItemDataNode(transform_id)
                    for id2 in transformed_ids:
                        node = sh.GetItemDataNode(id2)
                        node.SetAndObserveTransformNodeID(transform_node.GetID())
                    # get original transform node
                    tf = slicer.util.getNode("DBS DTT")
                    transform_node.SetAndObserveTransformNodeID(tf.GetID())
                    #
                    break

    def apply_transformation_mer(self, transform_node: vtkMRMLTransformNode):

        dtt_node = slicer.util.getNode("LeadOR:DTT")

        # get parent transform node

        par_transf = dtt_node.GetParentTransformNode()

        if par_transf.GetParentTransformNode() is None:
            par_transf.SetAndObserveTransformNodeID(transform_node.GetID())
        else:
            par_transf.SetAndObserveTransformNodeID(None)

    def on_calculate_shift(self):
        try:
            a = slicer.util.getNode('shift')
            slicer.mrmlScene.RemoveNode(a)
            print("UNSHIFTED")
            return
        except:
            pass
        self.onTextModified(self.text)
        self._remove_all_previous_copies()
        # self._copy_leadORIGTL_folder()

        # self.copy_leador_electrodes()
        try:
            self.apply_transformation_mer(slicer.util.getNode('shift'))
        except:
            pass
        try:
            el_nodes = slicer.util.getNode("Electrodes")
            el_nodes.SetAndObserveTransformNodeID(slicer.util.getNode('shift').GetID())
        except Exception as e:
            print(e)
            pass
        print("SHIFTED")
        # Compute the shift

        # self.logic.shift_estimation(self.left_e_rec, self.right_e_rec, self.pts_left, self.pts_right)
        pass

    def apply_transformation_markups(self, param):
        pass


def compute_transformation_from_signal(records,
                                       mesh_pts,
                                       min_max_pts,
                                       transformer, model):
    """

    """
    transl_scaller = MinMaxScaler().fit([[-10, -10, -10], [10, 10, 10]])  # config tesy
    m2 = np.reshape(mesh_pts, (-1, 1))
    m2 = m2.reshape(-1).tolist()

    # print(m2.shape, m2)
    pcas = transformer.transform([m2])
    # print()
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
    return transformNode


def slicer_transform_electrode(side):
    transform_name = "right_shift" if side == "right" else "left_shift"
    transform = slicer.util.getNode(transform_name)


def loadNiiImage(file_path):
    # Load an image and display it in Slicer
    image_node = slicer.util.loadVolume(file_path)
    slicer.util.setSliceViewerLayers(background=image_node)
    return image_node


#
# DBSShiftPredictionLogic
#
MESH_results = Tuple[vtkMRMLModelNode, np.ndarray]


def convert_to_vtk_matrix(np_matrix: np.ndarray) -> vtk.vtkMatrix4x4:
    """
    Convert a numpy matrix to a vtk matrix.
    """
    matrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            matrix.SetElement(i, j, np_matrix[i, j])
    return matrix


class MRI_MERLogic(ScriptedLoadableModuleLogic):
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
        try:
            import torch
        except:
            slicer.util.pip_install('torch')
            import torch
        try:
            import torchio
        except ImportError:
            slicer.util.pip_install('torchio')
            import torchio
        ScriptedLoadableModuleLogic.__init__(self)

        self.net_shift = TransformerShiftPredictor(4, 64, 1, 1, 10)
        cd_state_dict = torch.load(self.resourcePath('nets/net_transformer.pt'), map_location=torch.device('cpu'))
        self.net_shift.load_state_dict(cd_state_dict)

        self.mer_transforms = _read_pickle(self.resourcePath('nets/mer_pca_mesh.pkl'))

        self.mer_classification_net = TransformerClassifier(4, 64, 1, 1)
        cd_state_dict = torch.load(self.resourcePath('nets/transformer_classifier.pt'),
                                   map_location=torch.device('cpu'))
        self.mer_classification_net.load_state_dict(cd_state_dict)

    def read_leadOR_txt(self, text_node: vtkMRMLTextNode):
        """
        Read the text node and return the content.
        """
        from io import StringIO
        import pandas as pd
        text = text_node.GetText()
        iotext = StringIO(text)
        df = pd.read_csv(iotext)
        return df  # [1:]

    def nrms_df_calculation(self, df: pd.DataFrame) -> pd.DataFrame:

        # iterate through columns
        cols = []
        for col in df.columns:
            if not col.endswith('XYZ') and col != 'RecordingSiteDTT':
                cols.append(col)

        # compute means
        mean = df.iloc[1:5][cols].mean()
        print(mean)
        # compute nrms
        df[cols] = df[cols] / mean

        return df

    def nrms_min_max_scale(self, df: pd.DataFrame):
        cols = []
        for col in df.columns:
            if not col.endswith('XYZ') and col != 'RecordingSiteDTT':
                cols.append(col)

            # compute min max
        min_max = np.percentile(df[cols].values, 15), np.percentile(df[cols].values, 85)
        print(min_max)
        df.loc[:, cols] = (df.loc[:, cols] - min_max[0]) / (min_max[1] - min_max[0])
        df[cols] = df[cols].clip(0, 1)
        return df

    def remove_abnormals_first(self, df: pd.DataFrame):
        """
        set mean values if first recordings are abnormal (> mean)
        """
        cols = []
        for col in df.columns:
            if not col.endswith('XYZ') and col != 'RecordingSiteDTT':
                cols.append(col)

        # compute means
        mean = df[cols].mean()
        # print("MEAN", mean)
        # set to  if  value > mean ans
        test = df.loc[0:4, cols].where(df.loc[0:4, cols] < mean, np.nan)
        # print(test)
        # compute mean
        mean_f = test.mean()
        # set to mean values if > mean (mean for each column)
        for col in cols:
            df.loc[0:4, col] = df.loc[0:4, col].where(df.loc[0:4, col] < mean[col], mean_f[col])
        return df

    def df_to_electrode_records_tensor(self, df: pd.DataFrame, to_mni_transform: vtkMRMLTransformNode,
                                       require_mirror) -> Tuple[torch.Tensor, List[str], int]:
        """
        Convert a dataframe to a tensor of electrode records.
        """
        # Convert the dataframe to a tensor
        cols = [col for col in df.columns if col.endswith('XYZ')]

        to_mni = vtk.vtkMatrix4x4()
        to_mni_transform.GetMatrixTransformToWorld(to_mni)

        df[cols] = df[cols].applymap(lambda x: [float(a) for a in x.split(';')])
        # apply the transformation to each cell of df[cols]
        df[cols] = df[cols].applymap(lambda x: np.array(to_mni.MultiplyPoint(x + [1])[:3]))
        if require_mirror:
            df[cols] = df[cols].applymap(lambda x: x * np.array([-1, 1, 1]))
        cols = [col for col in df.columns if (not col.endswith('XYZ')) and (col != 'RecordingSiteDTT')]

        result_np = np.vstack(
            [np.array([np.concatenate((row[0], [row[1]])) for row in df[[x + "XYZ", x]].values]) for x in cols])

        tensor = torch.from_numpy(result_np).type(torch.float32)
        return tensor, cols, len(cols)

        # return tensor

    def electrode_records_rescale(self, electrode: torch.Tensor):
        """
        Rescale the electrode records.
        """
        min_max = self.mer_transforms['min_max_p']
        electrode[:, :3] = (electrode[:, :3] - min_max[0]) / (min_max[1] - min_max[0])
        return electrode

    def predict_initial_shift(self, electrode_tensor: torch.Tensor, mesh_pca_pts):
        """
        Predict the initial shift of the electrode.
        """
        # Apply the model to the electrode tensor
        mesh_tensor = torch.from_numpy(np.expand_dims(mesh_pca_pts, axis=0)).type(torch.float32)
        # print(mesh_tensor)
        # print(electrode_tensor)
        result = self.net_shift(electrode_tensor, mesh_tensor)
        return result

    #    def convert_df_to_electrode_records(self, df: pd.DataFrame, to_mni_transform, require_mirror) -> List[ElectrodeRecord]:

    def resourcePath(self, relativePath):
        """
        Get the absolute path to the module resource
        """
        # print("pt1", os.path.dirname(__file__))
        return os.path.normpath(os.path.join(os.path.dirname(__file__), "Resources", relativePath))

    def getParameterNode(self):
        return DBSShiftPredictionParameterNode(super().getParameterNode())

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

    def process_mer_data(self, left_markups,
                         right_markups,
                         to_mni,
                         mer_left_path,
                         mer_right_path):

        points_right = get_control_points(right_markups)
        points_left = get_control_points(left_markups)

        print(points_right)  # Print the positions of the control points for the right markups node
        print(points_left)  # Print the positions of the control points for the left markups node

        matrix = get_transform_matrix(transform_node=to_mni)
        numpy_array = convert_to_numpy_array(matrix)

        et_right = EntryTarget(Point.from_array(points_right[0]), Point.from_array(points_right[1]))
        et_left = EntryTarget(Point.from_array(points_left[0]), Point.from_array(points_left[1]))

        ea_left = apply_transformation(et_left, numpy_array)
        ea_right = apply_transformation(et_right, numpy_array)

        print(ea_right)  # Print the entry and target points for the right MER

        mer_left = read_mer_data(mer_left_path, "*run-02*")
        mer_right = read_mer_data(mer_right_path, "*run-01*")

        mer_left = process_mer_data(mer_left)
        mer_right = process_mer_data(mer_right)

        right_e_rec = extract_electrode_records(ea_right, mer_right, numpy_array)
        left_e_rec = extract_electrode_records(ea_left, mer_left, numpy_array)

        logic = LeadORLogic()
        i = 0
        for el_name, records in right_e_rec.items():
            print(records)
            logic.setUpTrajectory(i, getattr(ea_right, el_name), records, True, "right_" + el_name, 1, 1, 1)

        for el_name, records in left_e_rec.items():
            print(records)
            logic.setUpTrajectory(i, getattr(ea_left, el_name), records, True, "left_" + el_name, 1, 1, 1)

        return left_e_rec, right_e_rec

    def get_pcas_from_mesh(self, node: vtkMRMLModelNode, mirror=False):
        pts = extract_points_from_mesh(mesh=node.GetMesh())

        if mirror:
            pts = np.reshape(pts, (int(len(pts) / 3), 3))
            pts = pts * np.array([-1, 1, 1])
            pts = np.reshape(pts, (1, -1))[0]

            print(pts.shape)

        res_pts = self.mer_transforms['pipe'].transform([pts])[0]
        return res_pts, pts

    def classify_mers_clean(self, mer_data: torch.Tensor, number_of_electrodes: int):
        """
        clasify mers
        note shape of mer data is divisable by number of electrodes
        """
        classes = self.classify_mers(mer_data)

        # all first 4 recordings are set to 0
        num_records = classes.shape[0] // number_of_electrodes
        for el_i in range(number_of_electrodes):
            for pos in range(4):
                classes[el_i * num_records + pos] = 0

        # filter wrong classes
        for el_i in range(number_of_electrodes):

            for pos_of_recording in range(1, num_records - 1):

                # if class is 0 and previous and next is 1 then set to 1
                if classes[el_i * num_records + pos_of_recording] == 0 and classes[
                    el_i * num_records + pos_of_recording - 1] == 1 and classes[
                    el_i * num_records + pos_of_recording + 1] == 1:
                    classes[el_i * num_records + pos_of_recording][0] = 1
        return classes

    def classify_mers(self, mer_data: torch.Tensor):
        """
        clasify mers
        """

        with torch.no_grad():
            output = self.mer_classification_net(mer_data) > 0.5
        return output

    def _predict_shift(self, mer_data: torch.Tensor,
                       original_shift: torch.Tensor,
                       mesh: vtk.vtkPolyData, classes: torch.Tensor, lambda1=1.0, distance=0.2, lr=0.004):
        """
        predict shift
        input:
        mer_data: torch.Tensor unscalled mer data with shape (n, 4) x,y,z,record
        original_shift: torch.Tensor original estimated shift
        mesh: vtk.vtkPolyData mesh to be used for the optimisation
        """

        return optimise_mer_signal(mer_data, original_shift, mesh, classes, lambda1, distance,
                                   learning_rate=lr), classes

        pass

    def predict_shift(self, text_node: "vtkMRMLTextNode", side: bool,
                      to_mni: vtkMRMLLinearTransformNode, mesh: vtk.vtkPolyData, pcas, lambda1, distance,
                      learning_rate):

        df_text = self.read_leadOR_txt(text_node)

        df_text = df_text.drop_duplicates(subset=["RecordingSiteDTT"])

        df_text = self.remove_abnormals_first(df_text)

        df_text = self.nrms_df_calculation(df_text)

        df_text = self.nrms_min_max_scale(df_text)

        print(df_text[[x for x in df_text.columns if not x.endswith('XYZ') and x != 'RecordingSiteDTT']])

        result_tensor, mapping_columns, num_of_elec = self.df_to_electrode_records_tensor(df_text,
                                                                                          to_mni_transform=to_mni,
                                                                                          require_mirror=side)
        # note the result tensor is in mni scale and could be mirrored
        unscalled_tensor = result_tensor.clone()

        result_tensor = self.electrode_records_rescale(result_tensor)
        print(result_tensor.shape)

        shift = self.predict_initial_shift(result_tensor, pcas)

        shift = (shift * 20 - 10).detach()

        # markup_node = convert_tensor_to_markup_node(unscalled_tensor, side)
        classes = self.classify_mers_clean(unscalled_tensor, num_of_elec)
        (shift, scalling), classes = self._predict_shift(unscalled_tensor, torch.Tensor([0.0,0.0,0.0]),#shift,
                                                         mesh, classes, lambda1, distance,
                                                         learning_rate)

        text_node_res = compute_text(classes, df_text['RecordingSiteDTT'].values, mapping_columns)

        # logic.remove_previous_shift()
        converted_transform = self.convert_to_slicer_transformation(shift, scalling, to_mni,
                                                                    side)  # convert to slicer transformation

        return converted_transform, text_node_res

    def create_mesh_copy(self, node: vtkMRMLModelNode):
        """
        create a copy of the mesh
        return: vtkPolyData
        """

        # get mesh from node
        mesh = node.GetMesh()

        # create a copy of the mesh
        mesh_copy = vtk.vtkPolyData()
        mesh_copy.DeepCopy(mesh)
        return mesh_copy

        pass

    def convert_to_slicer_transformation(self, shift, scalling, to_mni: vtkMRMLTransformNode, require_mirror):
        """
        convert the shift to the slicer transformation from Native to Native
        """
        # get Matrix
        matrix = vtk.vtkMatrix4x4()
        to_mni.GetMatrixTransformToWorld(matrix)
        to_mni_array = convert_to_numpy_array(matrix)

        if require_mirror:
            mirr = np.eye(4)
            mirr[0, 0] = -1
            to_mni_array = np.dot(mirr, to_mni_array)

        # result shift     (to_mni^-1)*transl*to_mni

        shift_mat = np.eye(4)
        shift_mat[0, 0] = scalling[0]
        shift_mat[1, 1] = scalling[1]
        shift_mat[2, 2] = scalling[2]

        shift_mat2 = np.eye(4)
        shift_mat2[:3, 3] = (-shift)

        result_transform = np.dot(shift_mat, shift_mat2)
        tmp1 = np.dot(np.linalg.inv(to_mni_array), result_transform)
        result = np.dot(tmp1, to_mni_array)
        transformNode = vtkMRMLLinearTransformNode()
        transformNode.SetName("shift")
        transformNode.SetMatrixTransformToParent(convert_to_vtk_matrix(result))

        return transformNode
        pass

    def remove_previous_shift(self):
        """
        remove the previous shift node
        """
        try:
            shift_node = slicer.util.getNode("shift")
            slicer.mrmlScene.RemoveNode(shift_node)
        except:
            pass


#
# DBSShiftPredictionTest
#

class DBSShiftPredictionTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """

        self.logic = MRI_MERLogic()
        self.temp_workdir = tempfile.TemporaryDirectory()
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_load_and_coreg_images()
        self.test_bet()
        self.test_on_wm_segmentation()
        self.test_intensity_normalisation()
        self.test_two_step_coregistration()
        self.test_segmentation()
        self.test_mer_loading()
        self.test_mer_shift_estimation()
        # self.test_DBSShiftPrediction1()

    def tearDown(self):
        self.temp_workdir.cleanup()

    def test_load_and_coreg_images(self):
        """
        Test the loading of images
        """
        # load images
        path = Path(r"/home/varga/mounted_tuplak/processing_data/new_data_sorted/sub-P060")
        self.logic.processing_folder = path
        self.logic.on_chage_load_image("t1", "t1_precl_RAW.nii.gz", self.temp_workdir.name)
        t1 = slicer.util.getNode("t1")

        self.logic.on_chage_load_image("t2", "t2_precl_RAW.nii.gz", self.temp_workdir.name)
        t2 = slicer.util.getNode("t2")

        self.logic.coregistration_t2_t1(t1.GetStorageNode(), t2.GetStorageNode(),
                                        self.temp_workdir.name + "/coreg_t2.nii.gz")

    def test_two_step_coregistration(self):
        t2_node = slicer.util.getNode("t2")
        transform_node = self.logic.two_step_coregistration(t2_node, self.temp_workdir.name)
        transform_node.SetName("to_mni")

    def test_mer_loading(self):
        # create markup node for the MER
        right_markups = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode')
        left_markups = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode')
        # set points to markups
        right_markups.AddControlPointWorld(-35.5, 82.28, 76.58)
        right_markups.AddControlPointWorld(-15.09, 44.56, -2.32)
        right_markups.SetName("right")
        left_markups.AddControlPointWorld(82.27, 76.58, 4.95)
        left_markups.AddControlPointWorld(4.95, 42.64, -1.34)
        left_markups.SetName("left")
        # load MER files
        self.left_e_rec, self.right_e_rec = self.logic.process_mer_data(left_markups,
                                                                        right_markups,
                                                                        slicer.util.getNode('to_mni'),
                                                                        r"/home/varga/mounted_tuplak/mer_data_processing/mer/sub-P060/ses-perisurg/ieeg/sub-P060_ses-perisurg_run-01_channels.tsv",
                                                                        r"/home/varga/mounted_tuplak/mer_data_processing/mer/sub-P060/ses-perisurg/ieeg/sub-P060_ses-perisurg_run-01_channels.tsv")

    def test_DBSShiftPrediction1(self):
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
        inputVolume = SampleData.downloadSample('DBSShiftPrediction1')
        self.delayDisplay('Loaded test data set')

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = MRI_MERLogic()

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

    def test_intensity_normalisation(self):
        self.logic.intensity_normalisation(self.temp_workdir.name)
        slicer.mrmlScene.RemoveNode(slicer.util.getNode('t2'))
        t2_node = loadNiiImage(str(Path(self.temp_workdir.name) / "t2_normalised.nii.gz"))
        t2_node.SetName("t2")
        pass

    def test_on_wm_segmentation(self):
        slicer_preprocessing.wm_segmentation(t1=str(Path(self.temp_workdir.name) / "t1.nii.gz"),
                                             out_folder=self.temp_workdir.name)
        pass

    def test_bet(self):
        self.logic.brain_extraction(slicer.util.getNode('t1').GetStorageNode())
        slicer.mrmlScene.RemoveNode(slicer.util.getNode('t1'))
        t1_node = loadNiiImage(str(Path(self.temp_workdir.name) / "t1.nii.gz"))
        t1_node.SetName("t1")
        pass

    def test_segmentation(self):
        print("Starting segmentation")
        t2 = slicer.util.getNode("t2")

        left, right = self.logic.segmentSTNs(t2)
        self.pts_left = left[1]
        self.pts_right = right[1]
        pass

    def test_mer_shift_estimation(self):
        self.logic.shift_estimation(self.left_e_rec, self.right_e_rec, self.pts_left, self.pts_right)
        pass
