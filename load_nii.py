import nibabel as nib
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

def resize_image_with_crop_or_pad(image, img_size=(64, 64, 64)):   #source: https://f-i-tushar-eee.medium.com/3d-medical-imaging-pre-processing-all-you-need-6ba981738877
    """Image resizing. Resizes image by cropping or padding dimension
     to fit specified size.
    Args:
        image (np.ndarray): image to be resized
        img_size (list or tuple): new image size
        kwargs (): additional arguments to be passed to np.pad
    Returns:
        np.ndarray: resized image
    """

    assert isinstance(image, (np.ndarray, np.generic))
    assert (image.ndim - 1 == len(img_size) or image.ndim == len(img_size)), \
        'Example size doesnt fit image size'

    # Get the image dimensionality
    rank = len(img_size)

    # Create placeholders for the new shape
    from_indices = [[0, image.shape[dim]] for dim in range(rank)]
    to_padding = [[0, 0] for dim in range(rank)]

    slicer = [slice(None)] * rank

    # For each dimensions find whether it is supposed to be cropped or padded
    for i in range(rank):
        if image.shape[i] < img_size[i]:
            to_padding[i][0] = (img_size[i] - image.shape[i]) // 2
            to_padding[i][1] = img_size[i] - image.shape[i] - to_padding[i][0]
        else:
            from_indices[i][0] = int(np.floor((image.shape[i] - img_size[i]) / 2.))
            from_indices[i][1] = from_indices[i][0] + img_size[i]

        # Create slicer object to crop or leave each dimension
        slicer[i] = slice(int(from_indices[i][0]), int(from_indices[i][1]))

    # Pad the cropped image to extend the missing dimension
    return np.pad(image[tuple(slicer)], to_padding)


folder = "/home/azach/neverland/one_fold" 
final_shape = [256,256,256]

paths = sorted(glob.glob(os.path.join(folder, "sub*", "*.nii.gz"), recursive=True))
mri_paths = [p for p in paths if "T1w" in os.path.basename(p)]
mask_paths = [p for p in paths if "mask" in os.path.basename(p)]
if len(mri_paths) != len(mask_paths):
    raise ValueError("The number of MRI files does not match the number of mask files.")

stacked = np.zeros((len(mri_paths), 2, *final_shape))

for i, (mri_p,mask_p) in enumerate(zip(mri_paths, mask_paths)):
    mri= nib.load(mri_p).get_fdata()
    mask= nib.load(mask_p).get_fdata()
    mri_f=resize_image_with_crop_or_pad(mri, final_shape)
    stacked[i][0] = resize_image_with_crop_or_pad(mri, final_shape)
    stacked[i][1] = resize_image_with_crop_or_pad(mask, final_shape)

print("Stacked array shape:", stacked.shape)

"""
extent_or = [0, mri.shape[1], 0, mri.shape[0]]
f, axarr = plt.subplots(1, 2, figsize=(10,5));
#f, axarr = plt.subplots(1, 2, figsize=(mri_f.shape[1] / 20, mri_f.shape[0] / 30));
axarr[0].imshow(np.squeeze(mri[100, :, :]), cmap='gray',origin='lower', aspect='auto', extent=extent_or);
axarr[0].axis('on')
axarr[0].set_aspect(mri.shape[1] / mri.shape[2])
axarr[0].set_title('Original image {}'.format(mri.shape))

extent_pad = [0, mri_f.shape[1], 0, mri_f.shape[0]]
axarr[1].imshow(np.squeeze(mri_f[140, :, :]), cmap='gray',origin='lower', aspect='auto', extent=extent_pad);
axarr[1].axis('on')
axarr[0].set_aspect(mri_f.shape[1] / mri_f.shape[2])
axarr[1].set_title('Padded to {}'.format(mri_f.shape))

plt.tight_layout()
plt.subplots_adjust(wspace=0.2, hspace=0)
plt.show()
f.savefig('/home/azach/neverland/repository/3dMRISegmentation/figure.jpg', dpi=300, bbox_inches='tight', pad_inches=0.2, format='jpg')
"""
plt.figure(figsize=(6, 6))
plt.imshow(mri_f[140, :, :], cmap="gray", origin="lower", aspect='auto')
plt.title('Padded to {}'.format(mri_f.shape))
plt.axis("on")
plt.tight_layout()
plt.show()
plt.savefig('/home/azach/neverland/repository/3dMRISegmentation/figure.jpg', dpi=300, bbox_inches="tight", format="jpg")