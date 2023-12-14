import os
#
from pathlib import Path
import subprocess
import shlex
import nibabel as nib
import numpy as np
from intensity_normalization.cli.fcm import fcm_main
import sys


def register_t2_to_t1(script_home, t1, t2, out_name):
    parent = str(Path(out_name).parent)
    print(1)

    #
    flirt_command = (f"flirt -in {t2} "
                     f"-ref {t1} -out {out_name} ")
    flirt_command = shlex.split(flirt_command)

    subprocess.check_output(flirt_command)


def wm_segmentation(t1, out_folder):
    """
    t1: str t1 file
    out_folder: s
    """
    print(t1)

    copy_image = f"cp -f {t1} {str(Path(out_folder) / 't1.nii.gz')}"
    copy_image = shlex.split(copy_image)
    subprocess.check_output(copy_image)

    fast_command = f"fast -R 0.0 -H 0.0 -t 1 {str(Path(out_folder) / 't1.nii.gz')}"
    fast_command = shlex.split(fast_command)
    subprocess.check_output(fast_command)

    print(fast_command)


def binarise_threshold(filename, threshold, save_filename):
    # Load the NIfTI image
    nifti_img = nib.load(filename)

    # Get the image data as a NumPy array
    img_data = nifti_img.get_fdata()

    # Replace all occurrences of the old value with the new value
    img_data = img_data > threshold

    # Create a new NIfTI image from the modified data
    new_nifti_img = nib.Nifti1Image(img_data, nifti_img.affine, nifti_img.header)

    # Save the new NIfTI image to file
    nib.save(new_nifti_img, save_filename)


def intensity_normalisation(out_folder):
    """
    Script for intensity normalisation
    """
    file_name = "t2_normalised.nii.gz"
    t2_file = Path(out_folder) / "coreg_t2.nii.gz"
    pve_seg = Path(out_folder) / "t1_pveseg.nii.gz"
    wm_mask = str(Path(out_folder) / "wm_mask.nii.gz")
    binarise_threshold(filename=str(pve_seg), threshold=2, save_filename=wm_mask)

    run_wm_slab_creation = "fcm-normalize {t2_file} " \
                           "-tm {wm_mask} " \
                           "-o {o_file} -mo t2 -v".format(t2_file=t2_file
                                                          , wm_mask=wm_mask,
                                                          o_file=Path(out_folder) / file_name
                                                          )

    sys.argv = shlex.split(run_wm_slab_creation)
    fcm_main()


def elastix_registration(ref_image,
                         flo_image,
                         elastix_parameters,
                         out_folder):


    flirt_command = f"elastix -f {ref_image} -m {str(flo_image)} -p {elastix_parameters}" \
                    f" -out {out_folder}"
    flirt_command = shlex.split(flirt_command)
    subprocess.check_output(flirt_command)



def convert_matrix_to_slicer_transformation(matrix: np.ndarray, out_file: str):
    line = "# Insight Transform File V1.0\n"
    line += "Transform: AffineTransform_double_3_3\n"
    line += "Parameters: "
    line += " ".join([str(x) for x in matrix[:3, :3].flatten().tolist()])
    line += " " + " ".join([str(x) for x in matrix[:3, 3].tolist()]) + "\n"
    line += "FixedParameters: 0 0 0"

    o_f = open(out_file, "wt")
    o_f.write(line)
    o_f.close()
