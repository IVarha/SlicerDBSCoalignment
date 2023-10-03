from pathlib import Path





def get_images_in_folder(folder:str):
    fold = Path(folder)


    return [ x.name for x in fold.iterdir()
             if (x.name.endswith('nii.gz') or x.name.endswith('.nii'))

             ]


