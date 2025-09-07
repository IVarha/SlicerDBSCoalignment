import os
from pathlib import Path

import sm_scripts.file_utils as fi


######INPUTS AND OUTPUTS


WORKING_DIR=PROCESS_DIR = Path(config['process_dir'])

FSL_DIR="/usr/local/fsl"

#if exit tr
# MER_FOLDERS=""




rule all:
    input:
        f"{WORKING_DIR}/{config['output_file']}"

rule linear_registration:
    input:
        config['reference'],
        config['floating']
    output:
        f"{WORKING_DIR}/{config['output_file']}"

    run:
        #reg to mni
        flirt_command = f"flirt -in {input[1]} -ref {input[0]} -out"


        flirt_command = "flirt -in {struct_image} -ref {refim} -out {res_im}" \
                      " -omat {o_mat} -dof 12" \
            .format(struct_image= str(curr_dir/ "t1_acpc_extracted.nii.gz"),
            refim=FSL_MNI_TEMPLATE,res_im=str(curr_dir / "t1_brain_to_mni_affine"),
            o_mat =str( curr_dir / "affine_t1.mat" ))
        shell(flirt_command)
        #2nd stage
        flirt_command = "flirt -in {struct_image} -ref {refim} -out {res_im}" \
                      " -omat {o_mat} -nosearch -refweight {ref_weight}" \
                        ""\
            .format(struct_image=str(curr_dir/ "t1_brain_to_mni_affine.nii.gz") ,
            refim=FSL_MNI_TEMPLATE,res_im=str(curr_dir/ "t1_brain_to_mni_stage2"),
            o_mat =str(curr_dir/ "affine_t1_stage2.mat"), ref_weight = FSL_MNI_SC_MASK_WEIGHT )
        shell(flirt_command)
        # Convert matrices
        c_xfm = "convert_xfm -omat {omat} -concat {first_mat} {second_mat}"\
            .format(omat=str(curr_dir/ "combined_affine_t1.mat")  ,
            first_mat = str(curr_dir/ "affine_t1_stage2.mat")
            , second_mat= str(curr_dir/ "affine_t1.mat")   )
        shell(c_xfm)
        c_xfm = "convert_xfm -omat {omat} -inverse {first_mat}" \
            .format(omat=str(curr_dir/ "combined_affine_reverse.mat") ,
            first_mat=str(curr_dir/ "combined_affine_t1.mat") )
        shell(c_xfm)
        # apply transform
        flirt_command = "flirt -in {struct_image} -ref {refim} -out {res_im}" \
                      " -applyxfm -init {o_mat} -dof 12" \
            .format(struct_image=str(curr_dir/ "t1_acpc_extracted.nii.gz") ,
            refim=FSL_MNI_TEMPLATE,res_im=str(curr_dir/ "t1_brain_to_mni_stage2_apply") ,
            o_mat = str(curr_dir/ "combined_affine_t1.mat")  )
        shell(flirt_command)
        shell("touch " + str(curr_dir / "3_linear_registration"))

rule wm_segmentation:
    input:
        "{dirname}/3_linear_registration"
    output:
        "{dirname}/3_wm_seg_complete"
    threads: 1


    run:
        curr_dir = pathlib.Path(input[0]).parent

        name_1 = "t1_acpc_extracted.nii.gz"
        pveseg="t1_acpc_extracted_pveseg.nii.gz"
        fast_command = "fast -R 0.0 -H 0.0 {image}".format(image=str(curr_dir /  name_1))
        shell(fast_command)

        shell("touch " + str(curr_dir / "3_wm_seg_complete"))
        # for file1 in ac_pc_files:
            #     file_name = str(curr_dir/ file1.split(".")[0])
            #     print(file1)
            #     run_wm_slab_creation = "python {scr_dir}/wm_mask_slab.py " \
            #                            "{t2_file} {pve_seg} " \
            #                            "{out_wm_mask} " \
            #                            "{out_wm_mask2} " \
            #                            "{out_wm_mask3}".format(t2_file=str(curr_dir / file1),pve_seg = str(curr_dir / pveseg) ,
            #                                 out_wm_mask=file_name+"_WM_mask.nii.gz", out_wm_mask2= file_name + "_mask.nii.gz",
            #                                 out_wm_mask3=file_name+ "_brain_mask.nii.gz",scr_dir=SCRIPTS_DIR)
            #     print(run_wm_slab_creation)
            #     shell(run_wm_slab_creation)

# rule t2_bfc:
#     input:
#         INPUT_FOLDER / "3_wm_seg_complete"
#     output:
#         INPUT_FOLDER / "3_t2_bfc"
#     run:
#         for fold in os.listdir(str(INPUT_FOLDER)):
#             print(fold)
#             curr_dir =INPUT_FOLDER / fold
#             if not curr_dir.is_dir():
#                 continue
#             name_1 = "t1_acpc_extracted.nii.gz"
#             pveseg="t1_acpc_extracted_pveseg.nii.gz"
#
#             ac_pc_files = fi.end_with_and_contain(str(curr_dir),"_acpc.nii.gz","e1") \
#                           + fi.end_with_and_contain(str(curr_dir),"_acpc.nii.gz","e2") \
#                           + fi.end_with_and_contain(str(curr_dir),"_acpc.nii.gz","e3")
#
#             for file1 in ac_pc_files:
#
#                 file_name =str( curr_dir/file1.split(".")[0])
#
#                 bfc_command = "N4BiasFieldCorrection --image-dimensionality 3" \
#                               "  --input-image {struct_image} --output {resim} -x {br_mask} " \
#                               "".format(struct_image=file_name + ".nii.gz",
#                                 resim=file_name + "_bfc.nii.gz", br_mask=file_name+ "_brain_mask.nii.gz")
#                 # print(bfc_command)
#                 shell(bfc_command)
#         shell("touch " + str(INPUT_FOLDER / "3_t2_bfc"))

import nibabel as nib
import numpy as np
import SimpleITK as sitk
def replace_nifti_labels(filename, old_new_values, save_filename):
    # Load the NIfTI image
    nifti_img = sitk.ReadImage(filename)

    # Get the image data as a NumPy array
    img_data =  sitk.GetArrayFromImage(nifti_img)

    # Replace all occurrences of the old value with the new value
    a = []
    for el in old_new_values:
        tmp = (img_data == el[0])
        print(el[0])
        a.append(tmp.copy())
        #print(tmp.sum())
    for i in range(len(old_new_values)):

        img_data[a[i]] = old_new_values[i][1]
        print("a " + str(old_new_values[i][1]))
    print(img_data.sum())
    img_data = img_data.astype(np.float32)
    # Create a new NIfTI image from the modified data
    new_nifti_img = sitk.GetImageFromArray(img_data)
    new_nifti_img.CopyInformation(nifti_img)
    # Save the new NIfTI image to file
    sitk.WriteImage(new_nifti_img, save_filename)

def binarise_threshold(filename, threshold, save_filename):
    # Load the NIfTI image
    nifti_img = nib.load(filename)

    # Get the image data as a NumPy array
    img_data = nifti_img.get_fdata()

    # Replace all occurrences of the old value with the new value
    img_data= img_data > threshold

    # Create a new NIfTI image from the modified data
    new_nifti_img = nib.Nifti1Image(img_data, nifti_img.affine, nifti_img.header)

    # Save the new NIfTI image to file
    nib.save(new_nifti_img, save_filename)


rule intensity_normalisation:
    input:
        "{dirname}/3_wm_seg_complete"
    output:
        "{dirname}/3_intensity_normalisation"
    threads: 1
    run:

        print(input[0])
        curr_dir = pathlib.Path(input[0]).parent

        name_1 = "t1_acpc_extracted.nii.gz"
        pveseg="t1_acpc_extracted_pveseg.nii.gz"
        wm_mask = "t2_WM_mask.nii.gz"

        t2_name = "t2_raw.nii.gz"
        binarise_threshold(filename=curr_dir/pveseg,threshold=2
            ,save_filename=curr_dir/wm_mask)

        replace_nifti_labels(str(curr_dir/"labels.nii"),
            old_new_values=[(4,1)
                ,(1,2)
                ,(6,3),
                            (3,4),
                            (2,6)
                            ],
            save_filename=str(curr_dir/'labels2.nii'))
        shell( "gzip "  + str(curr_dir/'labels2.nii'))
        shell("mv " + str(curr_dir/'labels2.nii.gz') + " " + str(curr_dir/'labels.nii.gz'))
        file_name = "t2_normalised.nii.gz"
        run_wm_slab_creation = "fcm-normalize {t2_file} " \
                                   "-tm {wm_mask} " \
                                   "-o {o_file} -mo t2 -v".format(t2_file=curr_dir/t2_name
                                    ,wm_mask = curr_dir/wm_mask, o_file=curr_dir/file_name

                                    )
        shell(run_wm_slab_creation)


        shell("touch " + str(curr_dir / "3_intensity_normalisation"))

