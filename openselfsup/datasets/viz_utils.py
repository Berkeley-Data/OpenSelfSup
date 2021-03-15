import rasterio
import numpy as np
import matplotlib.pyplot as plt
import torch

# By default, it assumes that the data provided is imgae pixes and not the image file path
# Send 'is_pixel_data = False' if image paths are sent
def read_msi_as_plt(s1_data, s2_data, is_pixel_data = True):

    if not is_pixel_data:
        # Consider it as image file path
        imf = rasterio.open(s1_data)
        image_sequence = imf.read()
        # Convert to pixel data
        s1_data = np.array(image_sequence)

    if type(s1_data) == torch.Tensor:
        s1_data = s1_data.cpu().detach().numpy()

    print("s1_data shape ", s1_data.shape)
    ni = s1_data.shape[0]
    fig = plt.figure(figsize=(20, 15))

    for i in range(ni):
        plt.subplots_adjust(hspace=.2)
        plt.subplot(4, 5, i + 1)
        plt.imshow(s1_data[i], interpolation='nearest')
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.title('S1 - band ' + str(i + 1), fontsize=12, color='black')

    # Process S2 data
    s2_bands = [2, 3, 4, 5, 6, 7, 8, 9, 12, 13]
    if not is_pixel_data:
        # Consider it as image file path
        imf = rasterio.open(s2_data)
        image_sequence = imf.read()
        # Convert to pixel data
        s2_data = np.array(image_sequence)

    if type(s2_data) == torch.Tensor:
        s2_data = s2_data.cpu().detach().numpy()

    print("s2_data shape ", s2_data.shape)
    ni = s2_data.shape[0]
    # For S2 bands used during processing [2, 3, 4, 5, 6, 7, 8, 9, 12, 13]
    for i in range(ni):
        plt.subplots_adjust(hspace=.2)
        plt.subplot(4, 5, 5 + (i + 1))
        cmap_data = None
        if is_pixel_data:
            # From pipeline (Bands 2,3,4. Indexes are 0,1,2)
            if i == 0:
                cmap_data = 'Reds'
            elif i == 1:
                cmap_data = 'Greens'
            elif i == 2:
                cmap_data = 'Blues'
        else:
            # From file (Bands 2,3,4. Indexes are 1,2,3)
            if i == 1:
                cmap_data = 'Reds'
            elif i == 2:
                cmap_data = 'Greens'
            elif i == 3:
                cmap_data = 'Blues'

        if cmap_data:
            plt.imshow(s2_data[i], interpolation='nearest', cmap = cmap_data)
        else:
            plt.imshow(s2_data[i], interpolation='nearest')
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        # If we read the images form files, for S2 total bands are 13. But from pipelines, they are only 10
        plt.title('S2 - band ' + str( s2_bands[i] if is_pixel_data else (i + 1)), fontsize=12, color='black')

    # S2 bands 4,3,2 correspond to RGB
    rgbArray = np.zeros((256, 256, 3), 'uint8')
    rgbArray[..., 0] = 255.0 * (s2_data[3] - s2_data[3].min()) / (s2_data[3].max() - s2_data[3].min())
    rgbArray[..., 1] = 255.0 * (s2_data[2] - s2_data[2].min()) / (s2_data[2].max() - s2_data[2].min())
    rgbArray[..., 2] = 255.0 * (s2_data[1] - s2_data[1].min()) / (s2_data[1].max() - s2_data[1].min())
    plt.subplots_adjust(hspace=.2)
    # If we read the images form files, for S2 total bands are 13. But from pipelines, they are only 10
    plt.subplot(4, 5, 16 if ni < 12 else 19)
    plt.imshow(rgbArray, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title('S2 - RGB', fontsize=12, color='black')

    return plt

