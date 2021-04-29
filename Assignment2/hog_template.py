import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog


def apply_filter_1D(img, kernel):
    img = img.astype(np.float)
    img_height, img_width = img.shape
    filtered_img_x = np.zeros((img_height, img_width))
    filtered_img_y = np.zeros((img_height, img_width))
    kernel_radius = int(np.floor(kernel.shape[0] / 2.))

    # Manual convolution requires kernel flipping
    kernel = kernel[::-1]

    for row in range(kernel_radius, img_height - kernel_radius):
        for col in range(kernel_radius, img_width - kernel_radius):
            image_patch_x = img[row, col - kernel_radius:col + kernel_radius + 1]
            image_patch_y = img[row - kernel_radius:row + kernel_radius + 1, col]

            conv_x = np.sum(image_patch_x*kernel)
            conv_y = np.sum(image_patch_y*kernel)

            filtered_img_x[row, col] = conv_x
            filtered_img_y[row, col] = conv_y

    return filtered_img_x, filtered_img_y


def hog_2D(img, block_size, cell_size, orientations=9):
    '''
    This function computes the HoG feature values for a given image
    and returns the normalized feature values and per cell HoG values.
    :param img: Input image
    :param block_size: cells per block
    :param cell_size: pixels per cell
    :param orientations: orientations per 180 degrees
    :return: normalized_blocks: normalized features for each block
             image_cell_hog: HoG magnitude values for each bin of each cell of the image. Shape: [Cell_per_row x Cell_per_column x orientations]
    '''

    # Convert image to float type
    img = img.astype(np.float)
    img_height, img_width = img.shape

    # Containers for x,y derivative
    f_x = np.zeros(img.shape)
    f_y = np.zeros(img.shape)

    kernel = # 1D derivative kernel. You can use the central difference kernel here
    f_x, f_y = apply_filter_1D(img, kernel)

    # Get Magnitude
    mag = # Use hypot or manual sqrt to get magnitude

    # Get orientation of gradients, convert to degrees
    phase = # Use arctan2 and convert to degrees

    # Convert negative angles to equivalent positive angles, so that it has same direction
    phase[phase<0] += 180   # converts the negative angles (0 to -179) to corresponding positive angle [-20 is equivalent to +160]
    # phase = phase % 180   # Alternative way to convert
    phase[phase==180] = 0   # Make 180 as 0 for computation simplicity later

    # Calculate total number of cell rows and columns
    # Notice that it uses integer number of cell row,cols
    # If the image is of irregular size, we only compute till last position with full cell. If there are some pixels left which dont fill a full cell, it is ignored
    # Alternatively, you can also reshape the image to have height,width be divisible by pixels_per_cell.
    cell_rows = img_height // cell_size
    cell_cols = img_width // cell_size

    # Create container for HoG values per orientation for each cell.
    image_cell_hog = np.zeros((cell_rows, cell_cols, orientations))

    # Compute the angle each bin will have. For orientation 9, it should have 180/9 = 20 degrees per bin
    angle_per_bin = 180./orientations

    # This is the main HoG values computation part
    # Follow algorithm from class
    # Go through each cell
    for row in range(cell_rows):
        for col in range(cell_cols):
            # Each cell has N x N pixels in it. So get the patch of pixels for each cell
            # Get the magnitude and orientation patch
            cell_patch_mag = mag[row*cell_size:(row+1)*cell_size, col*cell_size:(col+1)*cell_size]
            cell_patch_orient = phase[] # Same way to get patch for phase

            # Now for each cell patch, go through each pixel
            # Get the orientation and magnitude
            # Find the bin based on orientation
            # Then add magnitude (weighted) to the bin(s)
            for rr in range(cell_size):
                for cc in range(cell_size):
                    # Get the current pixel's orientation and magnitude
                    current_orientation = # Get from orientation patch
                    current_magnitude = # Get from magnitude patch

                    # Find current bin based on magnitude
                    current_bin = # get bin by dividing orientation by angle per bin

                    # Use voting scheme from class
                    # Find what percentage of magnitude goes to bin on left and right of orientation
                    # So if orientation is 25, then it is between 20 and 40 and the current bin is 1
                    # But orientation of 25 means the magnitude should go somewhat to the bin for 40
                    # So a weighted value is assigned to both bins
                    # Find what percentage is to previous bin => 25 - 20 = 5. Then 5/20 = 25%.
                    # This means 25% of the magnitude should go to bin 2 and 75% should go to bin 1 as 25 is closer to bin 1
                    bin_left_percent = # Percent on bin to left
                    bin_left_value = # Find bin value for left bin (if you order bins left to right as 0,1,2,...,8 for [0,20,40,...,160] degrees
                    bin_right_value = # Find bin value for right bin

                    if current_bin+1 == orientations:  # last bin at 160, which will wrap around to 0 again
                        image_cell_hog[] += bin_left_value  # Add to current bin
                        image_cell_hog[] += bin_right_value # Add to bin 0 since it goes around
                    else:
                        image_cell_hog[] += bin_left_value  # Add to current bin
                        image_cell_hog[] += bin_right_value # Add to current bin + 1


    # Now normalize values per block
    # Find number of blocks for given cells per block that fits in the image.
    block_rows = cell_rows - block_size + 1
    block_cols = cell_cols - block_size + 1

    # Create container for features per block
    normalized_blocks = np.zeros((block_rows, block_cols, block_size*block_size*orientations))

    # Iterate through each block, get HoG values of cells in that block, normalize using L2 method
    for row in range(block_rows):
        for col in range(block_cols):
            # Get current block patch with given cells_per_block from image_cell_hog
            current_block_patch = # image_cell_hog[]

            # Normalize using L2 method.
            # Square each value, sum all of them, take square root
            normalized_block_patch = # Perform L2 normalization

            # Reshape to 1D array, gives [orientation * number of cells X 1]shape for each block
            normalized_block_patch = np.reshape(normalized_block_patch, -1)     # Make 1D

            # Assign the patch output to container
            normalized_blocks[row,col, :] = normalized_block_patch

    return normalized_blocks, image_cell_hog


if __name__=="__main__":
    # Read image as grayscale
    img = cv2.imread('canny1.jpg',0)

    # Set parameters
    block_size = 2      # Cells per block
    cell_size = 8       # Pixels per cell
    orientations = 9    # Orientations per 180 degrees

    '''
    Manual function to get HoG features.
    Takes image, block size, cell size and orientations.
    Returns the normalized blocks (HoG features per block) and the HoG values per cell of the image.
    For visualization, use the HoG values per cell (image_cell_hog) of shape [cells_per_row, cells_per_col]    
    '''
    normalized_blocks, image_cell_hog = hog_2D(img, block_size,cell_size, orientations)

    # For color coding, take the HoG values per cell, find maximum HoG value among all orientations.
    # Then normalize the max value and resize it to match image.
    hog_max = #Get max of image_cell_hog using numpy (only do it along last axis which has orientation).
    hog_max = hog_max/np.max(hog_max)
    hog_max = cv2.resize(hog_max, (img.shape[1],img.shape[0]), cv2.INTER_NEAREST)

    # Here you can implement your own method to draw a line along the bin with highest value.
    # For this option, you have to check which bin has highest value, then assign the highest value to the line
    # Finally, you will normalize the image so that unimportant lines will have lower weight.
    hog_image_2 = # Manual function to get lines along highest HoG orientation bin
    hog_image_2 = (hog_image_2 / np.max(hog_image_2) * 255).astype(np.uint8)

    # This is only for comparing your output with actual HoG output.
    # Do not use this for final submission. Only use this as reference
    features, scikit_hog_image = hog(img, orientations=orientations, pixels_per_cell=(cell_size, cell_size), cells_per_block=(block_size, block_size), visualize=True)
    scikit_hog_image = (scikit_hog_image/np.max(scikit_hog_image)*255).astype(np.uint8)

    # Now draw all the images/outputs using matplotlib.
    # Notice that the color is mapped to gray using cmap argument.
    # For final submission, remove scikit hog output. You can keep it blank.
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.suptitle('HoG')
    ax1.set_title('Image')
    ax1.imshow(img, cmap='gray')
    ax2.set_title("HoG color coding")
    ax2.imshow(hog_max, cmap='gray')
    ax3.set_title("Scikit HoG")
    ax3.imshow(scikit_hog_image, cmap='gray')
    ax4.set_title("Our HoG")
    ax4.imshow(hog_image_2, cmap='gray')
    plt.show()
