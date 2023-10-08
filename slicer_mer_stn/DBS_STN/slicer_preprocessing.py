import os
#
from pathlib import Path
import subprocess
import shlex


def register_t2_to_t1(script_home, t1, t2, out_name):
    parent = str(Path(out_name).parent)
    print(1)


    #
    flirt_command = (f"flirt -in {t2} "
                     f"-ref {t1} -out {out_name} ")
    flirt_command = shlex.split(flirt_command)

    subprocess.check_output(flirt_command)


def register_to_mni(t1, t2, out_name):
    parent = str(Path(out_name).parent)
    print(1)


    #
    flirt_command = (f"flirt -in {t2} "
                     f"-ref {t1} -out {out_name} ")
    flirt_command = shlex.split(flirt_command)

    subprocess.check_output(flirt_command)

