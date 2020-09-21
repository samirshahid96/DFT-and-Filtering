# For this part of the assignment, You can use inbuilt functions to compute the fourier transform
# You are welcome to use fft that are available in numpy and opencv
import numpy as np
import math


class Filtering:
    image = None
    filter = None
    cutoff = None
    order = None

    def __init__(self, image, filter_name, cutoff, order = 0):
        """initializes the variables frequency filtering on an input image
        takes as input:
        image: the input image
        filter_name: the name of the mask to use
        cutoff: the cutoff frequency of the filter
        order: the order of the filter (only for butterworth
        returns"""
        self.image = image
        if filter_name == 'ideal_l':
            self.filter = self.get_ideal_low_pass_filter
        elif filter_name == 'ideal_h':
            self.filter = self.get_ideal_high_pass_filter
        elif filter_name == 'butterworth_l':
            self.filter = self.get_butterworth_low_pass_filter
        elif filter_name == 'butterworth_h':
            self.filter = self.get_butterworth_high_pass_filter
        elif filter_name == 'gaussian_l':
            self.filter = self.get_gaussian_low_pass_filter
        elif filter_name == 'gaussian_h':
            self.filter = self.get_gaussian_high_pass_filter

        self.cutoff = cutoff
        self.order = order
        self.filter_name = filter_name
    " (-1)^i+j low in middle high in corbers"
    "D0 is the cutoff frequency"
    def get_ideal_low_pass_filter(self, shape, cutoff):
        """Computes a Ideal low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the ideal filter
        returns a ideal low pass mask"""
        P = shape[0]
        Q = shape[1]
        mask = np.zeros((P, Q))
        for u in range(P):
            for v in range(Q):
                dist = ((u - (P/2))**2 + (v - (Q/2))**2) ** (1/2)
                if dist > cutoff:
                    mask[u][v] = 0
                else:
                    mask[u][v] = 1

        return mask

    def get_ideal_high_pass_filter(self, shape, cutoff):
        """Computes a Ideal high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the ideal filter
        returns a ideal high pass mask"""

        #Hint: May be one can use the low pass filter function to get a high pass mask
        # 1 - ideal low pass
        mask = 1 - self.get_ideal_low_pass_filter(shape, cutoff)
        
        return mask

    def get_butterworth_low_pass_filter(self, shape, cutoff, order):
        """Computes a butterworth low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the butterworth filter
        order: the order of the butterworth filter
        returns a butterworth low pass mask"""
        P = shape[0]
        Q = shape[1]
        mask = np.zeros((P, Q))
        for u in range(P):
            for v in range(Q):
                dist = ((u - (P / 2)) ** 2 + (v - (Q / 2)) ** 2) ** (1 / 2)
                mask[u][v] = 1 / (1 + (dist/cutoff)**(2*order))

        return mask

    def get_butterworth_high_pass_filter(self, shape, cutoff, order):
        """Computes a butterworth high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the butterworth filter
        order: the order of the butterworth filter
        returns a butterworth high pass mask"""

        #Hint: May be one can use the low pass filter function to get a high pass mask
        # 1 - butter low pass
        mask = 1 - self.get_butterworth_low_pass_filter(shape, cutoff, order)

        return mask

    def get_gaussian_low_pass_filter(self, shape, cutoff):
        """Computes a gaussian low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the gaussian filter (sigma)
        returns a gaussian low pass mask"""
        P = shape[0]
        Q = shape[1]
        mask = np.zeros((P, Q))
        for u in range(P):
            for v in range(Q):
                dist = (((u - (P / 2)) ** 2) + ((v - (Q / 2)) ** 2)) ** (1 / 2)

                mask[u][v] = math.exp((-dist ** 2) / (2 * (cutoff ** 2)))

        return mask

    def get_gaussian_high_pass_filter(self, shape, cutoff):
        """Computes a gaussian high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the gaussian filter (sigma)
        returns a gaussian high pass mask"""

        #Hint: May be one can use the low pass filter function to get a high pass mask
        # 1 - gaussian low pass
        mask = 1 - self.get_gaussian_low_pass_filter(shape, cutoff)

        return mask

    def post_process_image(self, image):
        """Post process the image to create a full contrast stretch of the image
        takes as input:
        image: the image obtained from the inverse fourier transform
        return an image with full contrast stretch
        -----------------------------------------------------
        1. Full contrast stretch (fsimage)
        2. take negative (255 - fsimage)
        """
        A = np.min(image)
        B = np.max(image)
        k = 255
        R, C = image.shape
        for i in range(R):
            for j in range(C):
                image[i][j] = (k / (B-A)) * (image[i][j] - A)
        average_image = np.average(image)
        if average_image > 50:
            return image.astype(dtype='uint8')
        else:
            return (255-image).astype(dtype='uint8')

    def filtering(self):
        """Performs frequency filtering on an input image
        returns a filtered image, magnitude of DFT, magnitude of filtered DFT        
        ----------------------------------------------------------
        You are allowed to used inbuilt functions to compute fft
        There are packages available in numpy as well as in opencv
        Steps:
        1. Compute the fft of the image
        2. shift the fft to center the low frequencies
        3. get the mask (write your code in functions provided above) the functions can be called by self.filter(shape, cutoff, order)
        4. filter the image frequency based on the mask (Convolution theorem)
        5. compute the inverse shift
        6. compute the inverse fourier transform
        7. compute the magnitude
        8. You will need to do a full contrast stretch on the magnitude and depending on the algorithm you may also need to
        take negative of the image to be able to view it (use post_process_image to write this code)
        Note: You do not have to do zero padding as discussed in class, the inbuilt functions takes care of that
        filtered image, magnitude of DFT, magnitude of filtered DFT: Make sure all images being returned have grey scale full contrast stretch and dtype=uint8 
        """
        # 1 ###########################################################################################################
        fft_image = np.fft.fft2(self.image)
        # 2 ###########################################################################################################
        fft_shift_image = np.fft.fftshift(fft_image)

        ###
        mag_dft = np.log(np.abs(fft_shift_image))
        mag_dft = (255 * (mag_dft / np.max(mag_dft))).astype(dtype='uint8')
        ###

        # 3 ###########################################################################################################
        if self.filter_name == 'butterworth_l' or self.filter_name == 'butterworth_h':
            mask = self.filter(fft_shift_image.shape, self.cutoff, self.order)
        else:
            mask = self.filter(fft_shift_image.shape, self.cutoff)
        # 4 ###########################################################################################################
        # multiply the dft (fft shift image) by the mask
        filtered_image = fft_shift_image * mask

        ###
        mag_filtered_image = mag_dft * mask
        ###

        # 5 ###########################################################################################################
        inverse_fft_shift_image = np.fft.ifftshift(filtered_image)
        # 6 ###########################################################################################################
        inverse_fft_image = np.fft.ifft2(inverse_fft_shift_image)
        # 7 ###########################################################################################################
        mag_image = np.zeros(inverse_fft_image.shape, dtype=complex)
        for i in range(inverse_fft_image.shape[0]):
            for j in range(inverse_fft_image.shape[1]):
                if inverse_fft_image[i][j] < 0:
                    mag_image[i][j] = -1 * inverse_fft_image[i][j]
                else:
                    mag_image[i][j] = inverse_fft_image[i][j]
        # magnitude of inverse fft is complete
        # 8 ###########################################################################################################
        full_contrast_image = self.post_process_image(mag_image)

        return [mag_dft, mag_filtered_image, full_contrast_image]
