from dataclasses import dataclass
from typing import Optional, List, Iterable, Dict

import numpy as np
import torch
import vtk
from mer_lib.data import MER_data


@dataclass
class Point:
    x: float
    y: float
    z: float

    def __add__(self, other):
        if isinstance(other, np.ndarray):
            other = Point.from_array(other)
        return Point(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        if isinstance(other, np.ndarray):
            other = Point.from_array(other)
        return Point(self.x - other.x, self.y - other.y, self.z - other.z)

    def __truediv__(self, other):
        return Point(self.x / other, self.y / other, self.z / other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Point(self.x * other, self.y * other, self.z * other)
        elif isinstance(other, Point):
            return Point(self.x * other.x, self.y * other.y, self.z * other.z)
        elif isinstance(other, (list, np.ndarray)):
            return Point(self.x * other[0], self.y * other[1], self.z * other[2])
        else:
            return NotImplemented

    def compute_normal_vector(self):
        pt = np.array([self.x, self.y, self.z])
        res = pt / np.linalg.norm(pt)
        return Point(res[0], res[1], res[2])

    def __rmul__(self, other):
        return self.__mul__(other)

    def to_array(self):
        return np.array([self.x, self.y, self.z])

    @staticmethod
    def from_array(array):
        return Point(array[0], array[1], array[2])

    def apply_transformation(self, a: np.ndarray):
        """
        a : 4x4 transformation or 3x3
        """
        if a.shape[0] == 4:
            res = a @ np.array([self.x, self.y, self.z, 1])
            self.x, self.y, self.z = res[0], res[1], res[2]
        else:
            res = a @ np.array([self.x, self.y, self.z])
            self.x, self.y, self.z = res[0], res[1], res[2]


@dataclass
class EntryTarget:
    entry: Point
    target: Point


@dataclass
class ElectrodeArray:
    cen: EntryTarget
    lat: EntryTarget
    med: EntryTarget
    ant: EntryTarget
    pos: EntryTarget

    def compute_mni_transformation(self, to_mni: np.ndarray, array_label, mirror=False):
        """
        array label is mm label of a signal
        """


def compute_ElectrodeArray(line: EntryTarget, transform=None) -> ElectrodeArray:
    """
    from line computes entry target points for all electrodes
    """

    def rotation_matrix(b: Point) -> np.ndarray:
        """
        rotation matrix from vector b to vector a   (a is [0,0,1])

        """
        b = b.to_array()
        a = np.array([0, 0, 1])
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        v = np.cross(a, b)
        s = np.linalg.norm(v)
        c = np.dot(a, b)
        skew_symmetric_v = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        rotation_matrix = np.eye(3) + skew_symmetric_v + np.dot(skew_symmetric_v, skew_symmetric_v) * (
                (1 - c) / (s ** 2))
        return rotation_matrix

    ##################################

    if transform is None:
        transform = np.eye(4)

    norm_vector = line.entry - line.target

    cen_l = [0, 0, 0]
    lat_l = [-2, 0, 0]
    med_l = [2, 0, 0]
    ant_l = [0, 0, 2]
    pos_l = [0, 0, -2]

    from_space = np.linalg.inv(rotation_matrix(norm_vector))  # rotation matrix from norm_vector to [0,0,1]

    cen_en = line.entry + np.dot(from_space, cen_l)
    # print(from_space)
    lat_en = line.entry + from_space @ lat_l
    med_en = line.entry + from_space @ med_l
    ant_en = line.entry + from_space @ ant_l
    pos_en = line.entry + from_space @ pos_l

    cen_ex = line.target + from_space @ cen_l
    lat_ex = line.target + from_space @ lat_l
    med_ex = line.target + from_space @ med_l
    ant_ex = line.target + from_space @ ant_l
    pos_ex = line.target + from_space @ pos_l

    return ElectrodeArray(cen=EntryTarget(cen_en, cen_ex),
                          lat=EntryTarget(lat_en, lat_ex),
                          med=EntryTarget(med_en, med_ex),
                          ant=EntryTarget(ant_en, ant_ex),
                          pos=EntryTarget(pos_en, pos_ex))


def compute_vector_direction(central_point: Point, target_point: Point, distance=2) -> (Point, Point):
    """
    computes vector direction from central point to target point
    """

    pt1 = target_point - central_point
    pt1 = pt1.compute_normal_vector()  # normalise vector of direction

    res_pt = pt1 * distance + central_point
    return res_pt, pt1


def cross_generation_mni(ent_tg_native: EntryTarget, to_mni):
    """
    from entry target in native space
    generate entry target in mni space
    return ElectrodeArray in native space of generated EntryTarget
    """
    cen_l = [0, 0, 0]
    lat_l = [-2, 0, 0]
    med_l = [2, 0, 0]
    ant_l = [0, 0, 2]
    pos_l = [0, 0, -2]

    from_mni = np.linalg.inv(to_mni)
    entry_copy = ent_tg_native.entry.to_array()
    entry_copy = Point.from_array(entry_copy)
    entry_copy.apply_transformation(to_mni)

    # generate points in MNI space
    lat_mni = entry_copy + np.array(lat_l)
    med_mni = entry_copy + np.array(med_l)
    ant_mni = entry_copy + np.array(ant_l)
    pos_mni = entry_copy + np.array(pos_l)

    # generate points in native space
    lat_mni.apply_transformation(from_mni)
    lat_native = lat_mni
    med_mni.apply_transformation(from_mni)
    med_native = med_mni
    ant_mni.apply_transformation(from_mni)
    ant_native = ant_mni
    pos_mni.apply_transformation(from_mni)
    pos_native = pos_mni

    # generate entry target in native space
    lat_entry, lat_v = compute_vector_direction(central_point=ent_tg_native.entry, target_point=lat_native, distance=2)
    lat_target = (2 * lat_v) + ent_tg_native.target

    med_entry, med_v = compute_vector_direction(central_point=ent_tg_native.entry, target_point=med_native, distance=2)
    med_target = (2 * med_v) + ent_tg_native.target

    ant_entry, ant_v = compute_vector_direction(central_point=ent_tg_native.entry, target_point=ant_native, distance=2)
    ant_target = (2 * ant_v) + ent_tg_native.target

    pos_entry, pos_v = compute_vector_direction(central_point=ent_tg_native.entry, target_point=pos_native, distance=2)
    pos_target = (2 * pos_v) + ent_tg_native.target

    return ElectrodeArray(cen=EntryTarget(Point.from_array(ent_tg_native.entry.to_array()),
                                          Point.from_array(ent_tg_native.target.to_array())),
                          lat=EntryTarget(lat_entry, lat_target),
                          med=EntryTarget(med_entry, med_target),
                          ant=EntryTarget(ant_entry, ant_target),
                          pos=EntryTarget(pos_entry, pos_target))


@dataclass
class ElectrodeRecord:
    """
    electrode contain [x,y,z,NRMS]
    """
    location: Point
    record: float  # NRMS value for now
    label: int  # 0-out 1-in

    def get_record_label(self) -> (np.ndarray, np.ndarray):
        """
        return p
        """
        return np.array([self.location.x, self.location.y, self.location.z, self.record]), np.array([self.label])

    @staticmethod
    def electrode_list_to_array(electrode_records: Iterable["ElectrodeRecord"]):
        record, target = [], []
        for el_rec in electrode_records:
            x, y = el_rec.get_record_label()
            record.append(x)
            target.append(y)
        # for i in range(len(record)):
        #     print( record[i], target[i])
        result = np.vstack(record), np.vstack(target)
        # for i in range(len(record)):
        # print( result[0][i], result[1][i])
        return result

    @staticmethod
    def extract_electrode_records_from_array(array: ElectrodeArray,
                                             mer_data: MER_data,
                                             transformation: Optional[np.ndarray]) -> Dict[
        str, List["ElectrodeRecord"]]:
        """
        Extracts electrode records from an array based on MER data and a transformation matrix.

        Args:
            array (ElectrodeArray): The electrode array.
            mer_data (MER_data): The MER data.
            transformation (Optional[np.ndarray]): The transformation matrix. Defaults to None.

        Returns:
            List[ElectrodeRecord]: The extracted electrode records.
        """
        if transformation is None:
            transformation = np.eye(4)

        result = {}
        dists = mer_data.get_anat_landmarks()[1]

        for i_distance in range((mer_data.extracted_features.shape[1])):
            for el_indx in range(mer_data.get_num_electrodes()):
                el_name = mer_data.get_electrode_name_by_index(el_indx)
                if el_name not in result:
                    result[el_name] = []

                ent_targ: EntryTarget = getattr(array, el_name)
                vector = ent_targ.target - ent_targ.entry

                norm = vector / np.linalg.norm(vector.to_array())

                res_pt = ent_targ.target + dists[i_distance] * norm
                res_pt.apply_transformation(transformation)

                er = ElectrodeRecord(res_pt,
                                     record=mer_data.extracted_features[el_indx][i_distance],
                                     label=0)
                result[el_name].append(er)
                # result.append(er)
        return result


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


def compute_distances_to_mesh(mesh_points: np.ndarray, points_to_compute: torch.Tensor) -> (np.ndarray, np.ndarray):
    res_vector, res_dists = [], []

    for i in points_to_compute.shape[0]:
        mp = mesh_points - points_to_compute[i]
        norms = np.linalg.norm(mp, axis=1)
        id = norms.tolist().index(min(norms))
        res_vector.append(mp[id] / norms[id])
        res_dists.append(norms[id])
    return np.array(res_dists), np.array(res_vector)


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
            if labeled_points[i] == 1:
                x.append(2)
            else:
                x.append(1)
        else:
            x.append(0)

    return torch.from_numpy(np.array(x))


def extract_points_from_mesh(mesh):
    # Get the vtkPoints object from the mesh
    points = mesh.GetPoints()

    # Initialize an empty list to store the points
    points_list = []

    # Iterate over all points in the vtkPoints object
    for i in range(points.GetNumberOfPoints()):
        pt = points.GetPoint(i)
        points_list.append(pt[0])
        points_list.append(pt[1])
        points_list.append(pt[2])

    # Return the list of points
    return points_list


def optimisation_criterion(orig_points, in_out, shift, mesh: vtk.vtkPolyData):
    """
    Calculate the criterion value for a given set of original points, in_out values, shift vector, and mesh.
    Args:
        orig_points (torch.Tensor): The original points.
        in_out (torch.Tensor): The in_out values.
        shift (torch.Tensor): The shift vector.
        mesh (cMesh): The mesh object.

    Returns:
        torch.Tensor: The criterion value.
    """

    orig_points = orig_points[:, :3]
    if isinstance(orig_points, np.ndarray):
        orig_points = torch.from_numpy(orig_points)
    # print(orig_points)
    new_points = orig_points - shift
    #    if isinstance(orig_points,np.ndarray):
    #        points_inside_posttrans = mesh.is_points_inside(new_points.tolist())
    #    else:
    points_inside_posttrans = check_points_inside_vtk_mesh(mesh, new_points.detach().numpy())

    weight = generate_correctly_placed_bitmap(in_out, points_inside_posttrans)
    # print(f'    {(weight>0).sum():.2f}')

    mp = extract_points_from_mesh(mesh)
    # print ("mesh pt 0", mp[:3])
    mesh_pts = np.reshape(mp, (len(mp) // 3, 3))

    mesh_pts = torch.from_numpy(mesh_pts)
    # print(torch.abs(distances_to_mesh(new_points,mesh_pts)))

    result_error = (weight * torch.abs(distances_to_mesh(new_points, mesh_pts))).sum()
    if (weight > 0).sum() == 0:
        result_error = 0
    else:
        result_error = result_error / (weight > 0).sum()

    result_error = result_error + ((weight > 0).sum()) * 0.1

    return result_error


def clasify_mers(mer_data: Dict[str, List[ElectrodeRecord]], model):
    """
    clasify mers
    """

    tmp = {}
    tmp_vector = []
    for k, v in mer_data.items():
        # print(k)
        tmp[k] = len(v)
        tmp_vector += v
    record, target = ElectrodeRecord.electrode_list_to_array(tmp_vector)

    record = torch.from_numpy(record).float()
    target = torch.from_numpy(target).float()
    with torch.no_grad():
        output = model(record)

    # convert back electrode records
    output = output.numpy()
    i = 0
    res_el_record = {}
    for k, v in tmp.items():
        electrode_output = output[i:i + v] > 0.5
        res_el_record[k] = []
        for j in range(v):
            tmp_vector[i + j].label = electrode_output[j]
            res_el_record[k].append(tmp_vector[i + j])

        i += v
    return res_el_record



