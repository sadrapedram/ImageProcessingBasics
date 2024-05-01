import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import streamlit as st
from PIL import Image

class filterClass:

    def __init__(self):
        self.folder_path = 'data/'
    


        
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



    def median_blur(self,image):
        pil_image = Image.open(image)
        np_image = np.array(pil_image)
        gray_image = cv.cvtColor(np_image, cv.COLOR_RGB2GRAY)
        filtered_image = cv.medianBlur(gray_image, ksize=15)
        return filtered_image
    

    def bilateral_filter(self,image):
        pil_image = Image.open(image)
        np_image = np.array(pil_image)
        gray_image = cv.cvtColor(np_image, cv.COLOR_RGB2GRAY)
        filtered_image = cv.bilateralFilter(gray_image, d=20, sigmaColor=80, sigmaSpace=10)
        return filtered_image
    

    def mean_filter(self, image):
        pil_image = Image.open(image)
        np_image = np.array(pil_image)
        gray_image = cv.cvtColor(np_image, cv.COLOR_RGB2GRAY)
        filtered_image = cv.blur(gray_image,(20,20))
        return filtered_image
    
    def _gaussian2d(self, sigma, fsize): #fsize(3,5)
        m = fsize[0]
        n = fsize[1]
        x=np.arange(-m/2,m/2+1)
        y=np.arange(-n/2,n/2+1)
        X,Y= np.meshgrid(x,y,sparse=True)
        g = np.exp(-((X**2 + Y**2)/(2.0*sigma**2)))
        sum = g.sum()
        gaussianfilter = g/sum

        # print(X,Y)
        # print(g)
        # print(sum)
        return gaussianfilter
    

    def apply_gaussian_filter(self, image, sigma, fsize):
        # Generate Gaussian filter
        gaussian_filter = self._gaussian2d(sigma, fsize)
        
        pil_image = Image.open(image)
        np_image = np.array(pil_image)
        gray_image_one_channel = cv.cvtColor(np_image,cv.COLOR_RGB2GRAY)

        # Pad the image to handle borders
        pad_height = fsize[0] // 2
        pad_width = fsize[1] // 2
        padded_image = np.pad(gray_image_one_channel, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
        # print(padded_image)


        # Initialize the filtered image
        filtered_image = np.zeros_like(gray_image_one_channel, dtype=float)
        
        # # Apply convolution
        height, width = gray_image_one_channel.shape
        for i in range(height):
            for j in range(width):
                patch = padded_image[i:i+fsize[0]+1, j:j+fsize[1]+1]
                filtered_image[i, j] = np.sum(patch * gaussian_filter)
        
        return filtered_image.astype(np.uint8)
    
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


    def otsu_thresholding(self,imagebinary):
        image = imagebinary
        threshold_range = range(np.max(image))
        print(threshold_range)
        criterias = np.array([self._compute_otsu_criteria(image, th) for th in threshold_range])
        print("this is criterias:{}".format(criterias))
        best_threshold = threshold_range[np.argmin(criterias)]
        print("best otsu Threshold: {}".format(best_threshold))
        binary = imagebinary
        binary[binary > best_threshold] = 255
        binary[binary <= best_threshold] = 0
        print("otsu_image_shape:{}".format(binary.shape))
        return binary.astype(np.uint8) 
    

    def _convolve2d(self, image, kernel):
        # Get the dimensions of the image and the kernel
        image_height, image_width = image.shape
        kernel_height, kernel_width = kernel.shape
        
        # Calculate the padding
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2
        
        # Pad the image
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')    
        # Initialize the output image
        convolved_image = np.zeros_like(image)    
        # Perform 2D convolution
        for i in range(image_height):
            for j in range(image_width):

                # Extract the region of interest
                region = padded_image[i:i+kernel_height, j:j+kernel_width]

                # Perform element-wise multiplication and sum
                convolved_pixel = np.sum(region * kernel)
                convolved_image[i, j] = convolved_pixel
        
        return convolved_image

    def custum_compute_sobel(self, image):
        # Sobel operators for computing gradients
        sobel_x = np.array([[-.1, 0, .1],
                            [-.2, 0, .2],
                            [-.1, 0, .1]])
        
        sobel_y = np.array([[-.1, -.2, -.1],
                            [ 0,  0,  0],
                            [ .1,  .2,  .1]])
        
        # Convolve image with Sobel operators
        gradient_x = self._convolve2d(image, sobel_x)
        gradient_y = self._convolve2d(image, sobel_y)
        
        # Compute magnitude of the gradient
        magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        magnitude = (255/magnitude.max())*magnitude 
        # print(magnitude)
        return magnitude.astype(np.uint8) 

    def open_cv_canny(self, image):
    # Convert the uploaded file to a PIL Image object
        pil_image = Image.open(image)
        # Convert PIL image to NumPy array
        np_image = np.array(pil_image)
        gray_image = cv.cvtColor(np_image, cv.COLOR_RGB2GRAY)
        edges = cv.Canny(gray_image, threshold1=30, threshold2=150)
        edges = (255/edges.max())*edges
        return edges
    


    def cv2_compute_sobel(self, image):
        # Sobel operators for computing gradients
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
        
        sobel_y = np.array([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]])
        
        # Convolve image with Sobel operators
        gradient_x = cv.filter2D(image, -1, sobel_x)
        gradient_y = cv.filter2D(image, -1, sobel_y)
        
        # Compute magnitude of the gradient
        magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        print(magnitude)
        return magnitude.astype(np.uint8)
    





#instance of the class
if "my_instance" not in st.session_state:
    st.session_state.my_instance = filterClass()

my_instance = st.session_state.my_instance

if "step1" not in st.session_state:
    st.session_state.step1=None

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file= None
if "form2_submit" not in st.session_state:
    st.session_state.form2_submit=None
form2_submit = st.session_state.form2_submit

if "gen_edge_button" not in st.session_state:
    st.session_state.gen_edge_button = False



with st.sidebar:
    st.header("denoise and Edge Filter Menu")


    st.session_state.uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    st.session_state.denoise = st.radio("what denoise filter do you want to apply:",("Gaussian Filter","Median Filter","Mean Filter","Bilateral Filter"))
    
    
    
    if st.session_state.uploaded_file and st.session_state.denoise is not None:
        if st.button("submit desoise"):
            if st.session_state.denoise == "Gaussian Filter":

                step1_result = my_instance.apply_gaussian_filter(image=st.session_state.uploaded_file,fsize=(6,6),sigma=5)

            elif st.session_state.denoise == "Median Filter":
                step1_result = my_instance.median_blur(st.session_state.uploaded_file)
            elif st.session_state.denoise == "Mean Filter":
                step1_result = my_instance.mean_filter(st.session_state.uploaded_file)
            elif st.session_state.denoise == "Bilateral Filter":
                step1_result = my_instance.bilateral_filter(st.session_state.uploaded_file)
            st.session_state.step1_result = step1_result
            st.session_state.step1 = True
    
    if st.session_state.uploaded_file and st.session_state.denoise and st.session_state.step1 is not None:
        st.write("edge form")

        st.session_state.edge = st.radio("what Edge filter do you want to apply:",("custom sobel","open_cv default sobel","Canny","otsu"))
        st.session_state.gen_edge_button = st.button("Gen Edge")
        



# with form1:
#     form1.write("denoise form")
#     st.session_state.uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
#     denoise = form1.radio("what denoise filter do you want to apply:",("Gaussian Filter","Median Filter","Mean Filter","Bilateral Filter"))
#     st.session_state.denoise = denoise
#     submit_button = form1.form_submit_button("load Image")

# main page columns
col1, col2, col3 = st.columns(3) 
with col1: 
    if st.session_state.uploaded_file is not None:
        # Display the uploaded image
        st.image(st.session_state.uploaded_file, caption='Uploaded Image', use_column_width=True)


with col2:
    if "step1_result" in st.session_state:
        st.image(st.session_state.step1_result, caption='denoise image', use_column_width=True)
        


#ToDo 
with col3:
    if st.session_state.gen_edge_button and st.session_state.step1_result is not None:
        if st.session_state.edge=="custom sobel":
            edge_image = my_instance.custum_compute_sobel(st.session_state.step1_result)
        if st.session_state.edge=="open_cv default sobel":
            edge_image = my_instance.cv2_compute_sobel(st.session_state.step1_result)
        if st.session_state.edge=="Canny":
            # edge_image = my_instance.(st.session_state.step1_result)
            # st.image(edge_image, caption='edge image', use_column_width=True)
            edge_image = my_instance.cv2_compute_sobel(st.session_state.step1_result)
        if st.session_state.edge=="otsu":
            edge_image = my_instance.otsu_thresholding(st.session_state.step1_result)
            # edge_image.shape
            len(edge_image)
            # edge_image
        st.image(edge_image, caption='edge image', use_column_width=True)
