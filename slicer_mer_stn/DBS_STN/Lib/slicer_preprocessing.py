#
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from intensity_normalization.cli.fcm import fcm_main


def _run_cli_wm_segmentation(t1, out_folder):
    pass
def wm_segmentation(t1, out_folder):
    """
    t1: str t1 file
    out_folder: s
    """
    print(t1)

    # copy_image = f"cp -f {t1} {str(Path(out_folder) / 't1.nii.gz')}"
    # copy_image = shlex.split(copy_image)
    # subprocess.check_output(copy_image)



    # run the deep_atropos segmentation algorithm of white matter
    if sys.platform == 'win32':
        import antspynet
        import ants
        t1_image = ants.image_read(t1)
        res = antspynet.deep_atropos(t1_image, verbose=True)
        si = res['segmentation_image']
        wm = (si == 3) or (si == 4) or (si == 5)

        wm_file = str(Path(out_folder) / "wm_mask.nii.gz")
        ants.image_write(wm, wm_file)
    elif sys.platform == 'darwin':
        res = None








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
    works with out_folder/coreg_t2.nii.gz
    """
    file_name = "t2_normalised.nii.gz"
    t2_file = Path(out_folder) / "coreg_t2.nii.gz"
    wm_mask = str(Path(out_folder) / "wm_mask.nii.gz")


    run_wm_slab_creation = ["fcm-normalize", str(t2_file),"-tm", wm_mask, "-o",str(Path(out_folder) / file_name), "-mo", "t2", "-v"]
    sys.argv = run_wm_slab_creation
    fcm_main()


def elastix_registration(ref_image,
                         flo_image,
                         elastix_parameters,
                         out_folder):

    flirt_command = ["elastix", "-f", ref_image, "-m", str(flo_image), "-p", elastix_parameters, "-out", out_folder]
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
