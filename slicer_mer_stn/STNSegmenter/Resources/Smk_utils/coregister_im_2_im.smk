import os
from pathlib import Path

import sm_scripts.file_utils as fi


######INPUTS AND OUTPUTS


WORKING_DIR = Path(config['process_dir'])
OUT_MAT = config['out_mat']

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


        flirt_command = (f"flirt -in {input[1]} "
                         f"-ref {input[0]} -out {output[0]} -omat {str(WORKING_DIR/OUT_MAT)}")
        shell(flirt_command)
