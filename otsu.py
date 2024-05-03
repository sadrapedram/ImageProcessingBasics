import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

class otsu:

    def __init__(self):
        self.folder_path = 'data/'

    # def read_images(self):
    #     plt.figure()
    #     file = os.listdir(self.folder_path)
    #     images=[]
    #     image = cv.imread(os.path.join(self.folder_path, file),cv.IMREAD_GRAYSCALE)
    #     normalized = self.Normalization(image)

    #     print("chaneel number:",image.shape)


    #     return normalized

    def _normalization(self,image):
        target_min = 0
        target_max = 255
        normalized_images = cv.normalize(image, None, alpha=target_max, beta=target_min, norm_type=cv.NORM_MINMAX)
        return normalized_images


    def read_images(self):
        plt.figure(figsize=(12,10))
        file_list = os.listdir(self.folder_path)
        images=[]
        for file in file_list: 
            if file.endswith('.jpg') or file.endswith('.png'):
                image = cv.imread(os.path.join(self.folder_path, file))
                normalized = self._normalization(image)
                print("chaneel number:",image.shape)
                normalized.shape
                print(normalized.shape)
                images.append(normalized)

        return images

    def show_images(self,image):
        if len(image[0].shape) == 2:
            plt.imshow(image)   
            plt.show()
        else:
            plt.imshow(image,cmap='gray')   
            plt.show()


    def gray_images(self,images):
        gray_images=[]
        for image in images:
            gray_image_one_channel = cv.cvtColor(image,cv.COLOR_RGB2GRAY)
            print("chaneel number:",gray_image_one_channel.shape)
            gray_images.append(gray_image_one_channel)

        return gray_images
            
    def gray_histogram(self,image):
        hist = cv.calcHist([image[0]],[0],None,[255],[0,255])
        plt.title("gray Histogram")
        plt.xlabel("Bins")
        plt.ylabel("number of pixels")
        plt.plot(hist)
        plt.show()




    # def otsu_thresholding(self, image):
    #     threshold, binary = cv.threshold(image[0], 0, 256, cv.THRESH_BINARY + cv.THRESH_OTSU)
    #     print("Otsu's threshold:", threshold)
    #     return binary   

    def otsu_thresholding(self,imagebinary):
        image = imagebinary[0]
        threshold_range = range(np.max(image))
        print(threshold_range)
        criterias = np.array([self._compute_otsu_criteria(image, th) for th in threshold_range])
        print("this is criterias:{}".format(criterias))
        best_threshold = threshold_range[np.argmin(criterias)]
        print("best otsu Threshold: {}".format(best_threshold))
        binary = imagebinary[0]
        binary[binary > best_threshold] = 255
        binary[binary <= best_threshold] = 0
        print("otsu_image_shape:{}".format(gray_image[0].shape))
        return binary
    
    def _compute_otsu_criteria(self,binary_image,threshold):
        image = binary_image
        thresholded_im = np.zeros(image.shape)
        thresholded_im[image >= threshold] = 1
        #compute weights
        nb_pixels = image.size
        nb_pixels1 = np.count_nonzero(thresholded_im)
        # print(nb_pixels1)
        # print(nb_pixels)

        weight1 = nb_pixels1 / nb_pixels
        weight0 = 1 - weight1
        # if weight1 == 0 or weight0 == 0:
        #     return np.inf
        
        # find all pixels belonging to each class
        val_pixels1 = image[thresholded_im == 1]
        val_pixels0 = image[thresholded_im == 0]

        # compute variance of these classes
        var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0
        var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0
        print("criteria weights from inner{}{}{}{}{}".format(weight0,var0,weight1,var1,weight0 * var0 + weight1 * var1))
        return weight0 * var0 + weight1 * var1


if __name__ == '__main__':
    functional_cv= otsu()
    rgb_img = functional_cv.read_images()

    functional_cv.show_images(rgb_img[0])
    gray_image = functional_cv.gray_images(rgb_img)
    gray_image[0].shape
    print("gray_image_shape_set_len:{}".format(gray_image[0].shape))
    otsu_image =functional_cv.otsu_thresholding(gray_image)
    functional_cv.show_images(otsu_image)