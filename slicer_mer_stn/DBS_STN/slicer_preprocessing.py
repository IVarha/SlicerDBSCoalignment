import os
#
from pathlib import Path
import subprocess
import shlex
import nibabel as nib
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
    binarise_threshold(filename=str(pve_seg),threshold=2,save_filename=wm_mask)


    run_wm_slab_creation = "fcm-normalize {t2_file} " \
                           "-tm {wm_mask} " \
                           "-o {o_file} -mo t2 -v".format(t2_file=t2_file
                                                          , wm_mask=wm_mask,
                                                          o_file=Path(out_folder) / file_name
                                                          )

    sys.argv = shlex.split(run_wm_slab_creation)
    fcm_main()


def two_step_linear_coregistration(mni_image,mni_mask, out_folder):



    struct_image = str(Path(out_folder) / "t1.nii.gz")


    flirt_command = "flirt -in {struct_image} -ref {refim} -out {res_im}" \
                    " -omat {o_mat} -dof 12" \
        .format(struct_image=struct_image,
                refim=mni_image, res_im=str(Path(out_folder)  / "t1_brain_to_mni_affine"),
                o_mat=str(Path(out_folder)  / "affine_t1.mat"))

    flirt_command = shlex.split(flirt_command)
    subprocess.check_output(flirt_command)
    # 2nd stage
    flirt_command = "flirt -in {struct_image} -ref {refim} -out {res_im}" \
                    " -omat {o_mat} -nosearch -refweight {ref_weight}" \
                    "" \
        .format(struct_image=str(Path(out_folder)  / "t1_brain_to_mni_affine.nii.gz"),
                refim=mni_image, res_im=str(Path(out_folder)  / "t1_brain_to_mni_stage2"),
                o_mat=str(Path(out_folder)  / "affine_t1_stage2.mat"), ref_weight=mni_mask)
    flirt_command = shlex.split(flirt_command)
    subprocess.check_output(flirt_command)
    # Convert matrices
    c_xfm = "convert_xfm -omat {omat} -concat {first_mat} {second_mat}" \
        .format(omat=str(Path(out_folder)  / "combined_affine_t1.mat"),
                first_mat=str(Path(out_folder)  / "affine_t1_stage2.mat")
                , second_mat=str(Path(out_folder)  / "affine_t1.mat"))

    c_xfm = shlex.split(c_xfm)
    subprocess.check_output(c_xfm)
    c_xfm = "convert_xfm -omat {omat} -inverse {first_mat}" \
        .format(omat=str(Path(out_folder)  / "combined_affine_reverse.mat"),
                first_mat=str(Path(out_folder)  / "combined_affine_t1.mat"))
    c_xfm = shlex.split(c_xfm)
    subprocess.check_output(c_xfm)
    # apply transform
    flirt_command = "flirt -in {struct_image} -ref {refim} -out {res_im}" \
                    " -applyxfm -init {o_mat} -dof 12" \
        .format(struct_image=str(Path(out_folder)  / "t1_acpc_extracted.nii.gz"),
                refim=mni_image, res_im=str(Path(out_folder)  / "t1_brain_to_mni_stage2_apply"),
                o_mat=str(Path(out_folder)  / "combined_affine_t1.mat"))

    pass