import os
import cv2
import numpy as np

from matplotlib import pyplot as plt
from patchify import patchify
from PIL import Image
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import random
from keras.utils import to_categorical
import gc


def load_and_patchify_images_mask(root_directory, patch_size):
    """
    Load images and masks from specified directories, crop them to ensure
    dimensions are divisible by the patch size, and divide them into smaller
    patches. Normalize image patches using Min-Max scaling.
    """
    scaler = MinMaxScaler()

    image_dataset = []
    mask_dataset = []

    # Collect all image and mask filenames
    image_files = []
    mask_files = []

    for path, subdirs, files in os.walk(root_directory):
        dirname = path.split(os.path.sep)[-1]
        if dirname == 'images':
            image_files.extend(os.path.join(path, f) for f in files if f.endswith(".jpg"))
        elif dirname == 'masks':
            mask_files.extend(os.path.join(path, f) for f in files if f.endswith(".png"))

    # Ensure image and mask files are sorted
    image_files = sorted(image_files)
    mask_files = sorted(mask_files)

    # Process images and masks
    for image_path, mask_path in zip(image_files, mask_files):
        # Load and process image
        image = cv2.imread(image_path, 1)
        SIZE_X = (image.shape[1] // patch_size) * patch_size
        SIZE_Y = (image.shape[0] // patch_size) * patch_size
        image = Image.fromarray(image).crop((0, 0, SIZE_X, SIZE_Y))
        image = np.array(image)
        print("Now patchifying image:", image_path)
        patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)

        # Loop through patches and flatten the patch structure
        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                single_patch_img = patches_img[i, j, :, :]  # Shape: (1, 256, 256, 3)
                single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
                image_dataset.append(single_patch_img[0])  # Flattening by selecting the first element

        # Load and process mask
        mask = cv2.imread(mask_path, 1)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        SIZE_X = (mask.shape[1] // patch_size) * patch_size
        SIZE_Y = (mask.shape[0] // patch_size) * patch_size
        mask = Image.fromarray(mask).crop((0, 0, SIZE_X, SIZE_Y))
        mask = np.array(mask)
        print("Now patchifying mask:", mask_path)
        patches_mask = patchify(mask, (patch_size, patch_size, 3), step=patch_size)

        # Loop through mask patches and flatten the patch structure
        for i in range(patches_mask.shape[0]):
            for j in range(patches_mask.shape[1]):
                single_patch_mask = patches_mask[i, j, :, :]  # Shape: (1, 256, 256, 3)
                mask_dataset.append(single_patch_mask[0])  # Flattening by selecting the first element

    # Convert lists to arrays
    image_dataset = np.array(image_dataset)
    mask_dataset = np.array(mask_dataset)

    # Clear the image_files and mask_files lists
    del image_files
    del mask_files
    # Trigger garbage collection
    gc.collect()

    return np.array(image_dataset), np.array(mask_dataset)



def plot_random_patchify_images_mask(image_dataset,mask_dataset,patch_size):

    """
        Select random image-mask pairs from the provided datasets and save them as
        individual image files in a specified output directory.

        Parameters:
        ----------
        image_dataset : list
            A list of image patches obtained from the load_and_patchify_images_mask function.

        mask_dataset : list
            A list of mask patches corresponding to the images, also obtained from the
            load_and_patchify_images_mask function.

        patch_size : int
            The size of the square patches, used for reshaping the images and masks.

        Returns:
        -------
        None

        Internal workings:
        -------------------
        1. Creates an output directory to save the images if it doesn't exist.
        2. Randomly selects a specified number of indices (10) from the image dataset.
        3. For each selected index, creates a new figure for plotting and adds
        subplots for the image and mask patches.
        4. Saves each plot as a PNG file in the specified output directory and
        closes the figure to free memory.
    """

    # Create a directory to save the images if it doesn't exist
    output_directory = './Sample_sanity_images'  # Specify your output directory
    os.makedirs(output_directory, exist_ok=True)

    # Select 10 random indices
    num_samples = 10
    selected_indices = random.sample(range(len(image_dataset)), num_samples)

    # Save each image-mask pair in individual files
    for index in selected_indices:
        plt.figure(figsize=(12, 6))

        # Plot the image patch
        plt.subplot(1, 2, 1)
        plt.imshow(np.reshape(image_dataset[index], (patch_size, patch_size, 3)))
        plt.title(f"Image Patch {index}")
        plt.axis('off')  # Hide axes

        # Plot the mask patch
        plt.subplot(1, 2, 2)
        plt.imshow(np.reshape(mask_dataset[index], (patch_size, patch_size, 3)))
        plt.title(f"Mask Patch {index}")
        plt.axis('off')  # Hide axes

        plt.tight_layout()
        
        # Save the figure
        output_file_path = os.path.join(output_directory, f"image_mask_patch_{index}.png")
        plt.savefig(output_file_path)  # Save the figure
        plt.close()  # Close the figure to free memory

    print(f"Cropped images with rspective mask save, samples are saved in {output_directory}.")



def rgb_to_2D_label(label):
    """
    Suply our labale masks as input in RGB format. 
    Replace pixels with specific RGB values ...
    """
    #Do the same for all RGB channels in each hex code to convert to RGB
    Building = '#3C1098'.lstrip('#')
    Building = np.array(tuple(int(Building[i:i+2], 16) for i in (0, 2, 4))) # 60, 16, 152

    Land = '#8429F6'.lstrip('#')
    Land = np.array(tuple(int(Land[i:i+2], 16) for i in (0, 2, 4))) #132, 41, 246

    Road = '#6EC1E4'.lstrip('#') 
    Road = np.array(tuple(int(Road[i:i+2], 16) for i in (0, 2, 4))) #110, 193, 228

    Vegetation =  'FEDD3A'.lstrip('#') 
    Vegetation = np.array(tuple(int(Vegetation[i:i+2], 16) for i in (0, 2, 4))) #254, 221, 58

    Water = 'E2A929'.lstrip('#') 
    Water = np.array(tuple(int(Water[i:i+2], 16) for i in (0, 2, 4))) #226, 169, 41

    Unlabeled = '#9B9B9B'.lstrip('#') 
    Unlabeled = np.array(tuple(int(Unlabeled[i:i+2], 16) for i in (0, 2, 4))) #155, 155, 155

    label_seg = np.zeros(label.shape,dtype=np.uint8)
    label_seg [np.all(label == Building,axis=-1)] = 0
    label_seg [np.all(label==Land,axis=-1)] = 1
    label_seg [np.all(label==Road,axis=-1)] = 2
    label_seg [np.all(label==Vegetation,axis=-1)] = 3
    label_seg [np.all(label==Water,axis=-1)] = 4
    label_seg [np.all(label==Unlabeled,axis=-1)] = 5
    
    label_seg = label_seg[:,:,0]  #Just take the first channel, no need for all 3 channels
    
    return label_seg


# root_directory = '/home/mglocadmin/Downloads/Semantic segmentation dataset/'
# patch_size = 256  # Example patch size
# image_dataset, mask_dataset = load_and_patchify_images_mask(root_directory, patch_size)
# print('image_dataset',image_dataset.shape)
# print('mask_dataset',mask_dataset.shape)
# plot_random_patchify_images_mask(image_dataset,mask_dataset,patch_size)


# labels = []
# for i in range(mask_dataset.shape[0]):
#     label = rgb_to_2D_label(mask_dataset[i])
#     labels.append(label)    

# labels = np.array(labels)   
# labels = np.expand_dims(labels, axis=3)

# image_number = random.randint(0, len(image_dataset))
# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# plt.imshow(image_dataset[image_number])
# plt.subplot(122)
# plt.imshow(labels[image_number][:,:,0])
# plt.show()
