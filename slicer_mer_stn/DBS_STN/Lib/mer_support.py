from dataclasses import dataclass
from typing import Optional, List, Iterable, Dict

import numpy as np
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
        elif isinstance(other,(list,np.ndarray)):
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


def compute_vector_direction(central_point: Point, target_point: Point,distance=2) -> (Point, Point):
    """
    computes vector direction from central point to target point
    """

    pt1 = target_point - central_point
    pt1 = pt1.compute_normal_vector() # normalise vector of direction

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

    return ElectrodeArray(cen=EntryTarget(Point.from_array(ent_tg_native.entry.to_array()), Point.from_array(ent_tg_native.target.to_array())),
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
    record : float # NRMS value for now
    label: int # 0-out 1-in

    def get_record_label(self) ->(np.ndarray,np.ndarray):
        """
        return p
        """
        return np.array([self.location.x,self.location.y,self.location.z,self.record]),np.array([self.label])

    @staticmethod
    def electrode_list_to_array( electrode_records : Iterable["ElectrodeRecord"]):
        record,target = [],[]
        for el_rec in electrode_records:
            x,y = el_rec.get_record_label()
            record.append(x)
            target.append(y)
        # for i in range(len(record)):
        #     print( record[i], target[i])
        result = np.vstack(record),np.vstack(target)
        #for i in range(len(record)):
             #print( result[0][i], result[1][i])
        return result

    @staticmethod
    def extract_electrode_records_from_array(array : ElectrodeArray,
                                             mer_data: MER_data,
                                             transformation: Optional[np.ndarray]) -> Dict[str,List["ElectrodeRecord"]] :
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

                ent_targ : EntryTarget = getattr(array,el_name)
                vector= ent_targ.target - ent_targ.entry

                norm = vector / np.linalg.norm(vector.to_array())


                res_pt = ent_targ.target + dists[i_distance] * norm
                res_pt.apply_transformation(transformation)

                er = ElectrodeRecord(res_pt,
                                     record=mer_data.extracted_features[el_indx][i_distance],
                                     label=0)
                result[el_name].append(er)
                #result.append(er)
        return result

