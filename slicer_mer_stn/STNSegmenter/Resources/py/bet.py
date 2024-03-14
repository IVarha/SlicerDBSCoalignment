


import sys
from pathlib import Path

import nibabel as nib
from brainextractor import BrainExtractor

image_t1 = sys.argv[1]
image = nib.load(image_t1)
bet = BrainExtractor(image)
bet.run()
print("FINISHED EXTRACTOR")
mask_filename= sys.argv[2]
bet.save_mask(mask_filename)

image = nib.load(image_t1)
mask = nib.load(mask_filename)

image_data = image.get_fdata()
mask_data = mask.get_fdata()
# Apply the mask to the image data
masked_image_data = image_data * mask_data

# Create a new NIfTI image from the masked data
masked_image = nib.Nifti1Image(masked_image_data, affine=image.affine)

out_image_name = sys.argv[3]

# Save the masked image
nib.save(masked_image, out_image_name)