import os
#
from pathlib import Path
import subprocess
import shlex
import nibabel as nib
import numpy as np
from intensity_normalization.cli import main as intensity_cli_main
import sys


def register_t2_to_t1(script_home, t1, t2, out_name):
    parent = str(Path(out_name).parent)
    print(1)

    #
    flirt_command = (f"flirt -in {t2} "
                     f"-ref {t1} -out {out_name} ")
    flirt_command = shlex.split(flirt_command)

    subprocess.check_output(flirt_command)

def _run_cli_wm_segmentation(t1, out_folder,pyth):
    import subprocess
    import shlex

    cmd = f"{pyth} -c 'import antspynet;import ants;from pathlib import Path;"
    cmd += f"t1_image = ants.image_read(\"{t1}\");"
    cmd += f"res=antspynet.deep_atropos(t1_image, verbose=True);si=res[\"segmentation_image\"];"
    cmd += f"wm=(si==3) or (si==4) or (si==5);"
    cmd += f"wm_file=str(Path(\"{out_folder}\") / \"wm_mask.nii.gz\");"
    cmd += f"ants.image_write(wm, wm_file)'"


    print(cmd)

    cmd = shlex.split(cmd)
    subprocess.check_output(cmd,env={})

    pass

def wm_segmentation(t1, out_folder,pyth=None):
    """
    t1: str t1 file
    out_folder: s
    """
    print(t1)

    import antstorch
    import ants
    t1_image = ants.image_read(t1)
    res = antstorch.deep_atropos([t1_image,None,None], verbose=True)
    si = res['segmentation_image']
    wm = (si == 3) or (si == 4) or (si == 5)

    wm_file = str(Path(out_folder) / "wm_mask.nii.gz")
    ants.image_write(wm, wm_file)





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


def intensity_normalisation(out_folder, t2_file):
    """
    Script for intensity normalisation
    works with out_folder/coreg_t2.nii.gz
    """
    file_name = "t2_normalised.nii.gz"
    output_path = Path(out_folder) / file_name
    t2_path = Path(t2_file)
    wm_mask_path = Path(out_folder) / "wm_mask.nii.gz"

    run_wm_slab_creation = [
        "intensity-normalization",
        "fcm",
        str(t2_path),
        "--output",
        str(output_path),
        "--modality",
        "t2",
        "--tissue-type",
        "wm",
        "--verbose",
    ]

    # Always provide a WM mask. If the mask grid differs from T2, resample it to T2 space
    # with nearest-neighbor interpolation to keep labels discrete.
    if wm_mask_path.exists():
        try:
            t2_shape = nib.load(str(t2_path)).shape[:3]
            wm_shape = nib.load(str(wm_mask_path)).shape[:3]
            print(f"Preparing WM mask for T2 normalization: T2 {t2_shape}, WM {wm_shape}")
            import ants

            t2_ants = ants.image_read(str(t2_path))
            wm_ants = ants.image_read(str(wm_mask_path))
            wm_resampled = ants.resample_image_to_target(
                wm_ants, t2_ants, interp_type="nearestNeighbor"
            )
            wm_resampled = (wm_resampled > 0.5)
            wm_resampled_path = Path(out_folder) / "wm_mask_resampled_to_t2.nii.gz"
            ants.image_write(wm_resampled, str(wm_resampled_path))
            run_wm_slab_creation.extend(["--mask", str(wm_resampled_path)])
        except Exception as exc:
            raise RuntimeError(
                f"WM mask exists but could not be aligned to T2 space: {exc}"
            ) from exc
    else:
        raise RuntimeError(
            f"Expected WM mask not found: {wm_mask_path}. Run WM segmentation first."
        )

    old_argv = sys.argv
    try:
        sys.argv = run_wm_slab_creation
        intensity_cli_main()
    finally:
        sys.argv = old_argv

    if not output_path.exists():
        raise RuntimeError(
            f"Intensity normalization did not produce output file: {output_path}"
        )



def elastix_registration_cmd(ref_image,
                         flo_image,
                         elastix_parameters,
                         out_folder):
    flirt_command = [ "-f", ref_image, "-m", str(flo_image), "-p", elastix_parameters, "-out", out_folder]
    return flirt_command


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
