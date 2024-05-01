#Image basics operations, and Image resizing and filtering.

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import streamlit as st
from PIL import Image





class imageProcessing:

    def __init__(self):
        self.folder_path = 'data/'

    def gray_images(self,image):
        pil_image = Image.open(image)
        np_image = np.array(pil_image)
        gray_image_one_channel = cv.cvtColor(np_image,cv.COLOR_RGB2GRAY)
        print("chaneel number:",gray_image_one_channel.shape)

        return gray_image_one_channel
    

    def color_histogram(self,image):
        pil_image = Image.open(image)
        np_image = np.array(pil_image)
        channels = cv.split(np_image)       # Set the image channels
        colors = ("b", "g", "r")        # Initialize tuple 
        plt.figure()    
        plt.title("Color Histogram")
        plt.xlabel("Bins")
        plt.ylabel("Number of Pixels")

        for (i, col) in zip(channels, colors):       # Loop over the image channels
            hist = cv.calcHist([i], [0], None, [256], [0, 256])   # Create a histogram for current channel
            plt.plot(hist, color = col)      # Plot the histogram
            plt.xlim([0, 256])
        return plt.gcf()

    def gray_histogram(self,image):
        pil_image = Image.open(image)
        np_image = np.array(pil_image)
        hist = cv.calcHist([np_image],[0],None,[256],[0,256])
        plt.title("gray Histogram")
        plt.xlabel("Bins")
        plt.ylabel("number of pixels")
        fig= plt.plot(hist)
        return plt.gcf() 
    
    def mean_filter(self, image,pixel):
        pil_image = Image.open(image)
        np_image = np.array(pil_image)
        gray_image = cv.cvtColor(np_image, cv.COLOR_RGB2GRAY)
        filtered_image = cv.blur(gray_image,(pixel,pixel))
        return filtered_image


    def bilateral_filter(self,image):
        pil_image = Image.open(image)
        np_image = np.array(pil_image)
        gray_image = cv.cvtColor(np_image, cv.COLOR_RGB2GRAY)
        filtered_image = cv.bilateralFilter(gray_image, d=20, sigmaColor=80, sigmaSpace=10)
        return filtered_image
    
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
    

    def open_cv_canny(self, image):
    # Convert the uploaded file to a PIL Image object
        pil_image = Image.open(image)
        # Convert PIL image to NumPy array
        np_image = np.array(pil_image)
        gray_image = cv.cvtColor(np_image, cv.COLOR_RGB2GRAY)
        edges = cv.Canny(gray_image, threshold1=30, threshold2=150)
        edges = (255/edges.max())*edges
        return edges
    

    def subtract_images(self,image1,image2):
        dsize = (1024,1024)

        pil_image = Image.open(image1)
        # Convert PIL image to NumPy array
        first_image = np.array(pil_image)
        first_image_resize = cv.resize(src=first_image,dsize=dsize)


        pil2_image = Image.open(image2)
        # Convert PIL image to NumPy array
        second_image = np.array(pil2_image)
        second_image_resize = cv.resize(src=second_image,dsize=dsize)

        subtract = cv.subtract(first_image_resize,second_image_resize)
        return subtract
    

    def sum_images(self,image1,image2):
        dsize = (1024,1024)
        pil_image = Image.open(image1)
        # Convert PIL image to NumPy array
        first_image = np.array(pil_image)
        first_image_resize = cv.resize(src=first_image,dsize=dsize)
        pil2_image = Image.open(image2)
        # Convert PIL image to NumPy array
        second_image = np.array(pil2_image)
        second_image_resize = cv.resize(src=second_image,dsize=dsize)
        sum = cv.add(first_image_resize,second_image_resize)
        return sum
    

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
    


# Streamlit session_state parameters
#instance of the class
if "my_instance" not in st.session_state:
    st.session_state.class_instance = imageProcessing()

if 'image' not in st.session_state:
    st.session_state['image'] = None

if 'image1' not in st.session_state:
    st.session_state['image1'] = None

if 'image2' not in st.session_state:
    st.session_state['image2'] = None

if 'sum_image' not in st.session_state:
    st.session_state['sum_image'] = None

if 'subtract_image' not in st.session_state:
    st.session_state['subtract_image'] = None

if 'image_result' not in st.session_state:
    st.session_state['image_result'] = None

if 'image_shift_to_two' not in st.session_state:
    st.session_state['image_shift_to_two'] = 1

if 'process_type' not in st.session_state:
    st.session_state['process_type'] = None

if 'fig' not in st.session_state:
    st.session_state['fig'] = None

if 'color_fig' not in st.session_state:
    st.session_state['color_fig'] = None

if 'blur_image' not in st.session_state:
    st.session_state['blur_image'] = None

if 'pixel' not in st.session_state:
    st.session_state['pixel'] = None

if 'edge_image_custom_sobel' not in st.session_state:
    st.session_state['edge_image_custom_sobel'] = None

if 'edge_image_canny' not in st.session_state:
    st.session_state['edge_image_canny'] = None

if 'difference_of_gaussian' not in st.session_state:
    st.session_state['difference_of_gaussian'] = None

    #DoG params
if 'first_sigma' not in st.session_state:
    st.session_state['first_sigma'] = None
if 'first_fsize' not in st.session_state:
    st.session_state['first_fsize'] = None
if 'second_sigma' not in st.session_state:
    st.session_state['second_sigma'] = None
if 'second_fsize' not in st.session_state:
    st.session_state['second_fsize'] = None
# side bar of the application

with st.sidebar:
    if st.session_state['image_shift_to_two']==1:
        if st.button("click to get two uploader"):
            st.session_state['image_shift_to_two']=2
            st.rerun()
        st.session_state['image'] =  st.file_uploader(label="upload image",type=['png','jpg','jpeg'])
        st.session_state['process_type'] = st.radio(label="choose your desired algorithm on the image:",
                options=['denoise image',
                        'Resizing',
                        'Rotating',
                        'Cropping',
                        'Flipping',
                        'Blurring',
                        'Thresholding',
                        'Converting Color Spaces',
                        'image gray',
                        'gray histogram',
                        'color histogram',
                        'edges of image',
                        'DoG'
                        ])
        if st.session_state['process_type'] == 'Blurring':
            st.session_state['pixel'] = st.slider(label='how many pixel width do you like to Blur',min_value=3,max_value=35)
        if st.session_state['process_type'] == 'DoG':
            st.session_state['first_sigma'] = st.slider(label='first sigma',min_value=0.0,max_value=3.0)
            st.session_state['first_fsize'] = st.slider(label='first fsize',min_value=3,max_value=35)
            st.session_state['second_sigma'] = st.slider(label='second sigma',min_value=0.0,max_value=3.0)
            st.session_state['second_fsize'] = st.slider(label='second fsize',min_value=3,max_value=35)

    elif st.session_state['image_shift_to_two']==2:
        if st.button("click to get one uploader"):
            st.session_state['image_shift_to_two']=1
            st.rerun()
        st.session_state['image1'] =  st.file_uploader(label="upload first image",type=['png','jpg','jpeg'])
        st.session_state['image2'] =  st.file_uploader(label="upload second image",type=['png','jpg','jpeg'])
        st.session_state['process_type'] = st.radio(label="choose your desired algorithm on the image:",
                options=[
                        'subtract two images',
                        'sum of two images'
                        ])
        
    # single image processes
    if st.button('process'):
        if st.session_state['image_shift_to_two']==1:
            if st.session_state['process_type'] == 'image gray':
                st.session_state['image_result'] = st.session_state.class_instance.gray_images(st.session_state['image'])
            elif st.session_state['process_type'] == 'gray histogram':
                st.session_state['fig'] = st.session_state.class_instance.gray_histogram(st.session_state['image'])
            elif st.session_state['process_type'] == 'color histogram':
                st.session_state['color_fig'] = st.session_state.class_instance.color_histogram(st.session_state['image'])
            elif st.session_state['process_type'] == 'Blurring':
                st.session_state['blur_image'] = st.session_state.class_instance.mean_filter(st.session_state['image'],pixel=st.session_state['pixel'])
            elif st.session_state['process_type'] == 'edges of image':
                denoised_image = st.session_state.class_instance.bilateral_filter(st.session_state['image'])
                st.session_state['edge_image_custom_sobel']  = st.session_state.class_instance.custum_compute_sobel(denoised_image)                                   
                st.session_state['edge_image_canny']  = st.session_state.class_instance.open_cv_canny(st.session_state['image'])                                   
            elif st.session_state['process_type'] == 'DoG':
                first_gaussian  = st.session_state.class_instance.apply_gaussian_filter(image=st.session_state['image'],sigma=st.session_state['first_sigma'],fsize=(st.session_state['first_fsize'],st.session_state['first_fsize']))   
                second_gaussian = st.session_state.class_instance.apply_gaussian_filter(image=st.session_state['image'],sigma=st.session_state['second_sigma'],fsize=(st.session_state['second_fsize'],st.session_state['second_fsize'])) 
                st.session_state['difference_of_gaussian']= cv.subtract(first_gaussian, second_gaussian)
        if st.session_state['image_shift_to_two']==2:
            if st.session_state['process_type'] == 'subtract two images':
                st.session_state['subtracted_image'] = st.session_state.class_instance.subtract_images(st.session_state['image1'],st.session_state['image2'])
            if st.session_state['process_type'] == 'sum of two images':
                st.session_state['sum_image'] = st.session_state.class_instance.sum_images(st.session_state['image1'],st.session_state['image2'])

col1, col2 = st.columns(2)


with col1:
    if st.session_state['image_shift_to_two']==2:
        if st.session_state['image1'] is not None:
            st.image(st.session_state['image1'])
        if st.session_state['image2'] is not None:
            st.image(st.session_state['image2'])

    elif st.session_state['image_shift_to_two']==1:
        if st.session_state['image'] is not None:
            st.image(st.session_state['image'])

with col2:
    if st.session_state['process_type'] is not None:
        if st.session_state['process_type'] == 'image gray':
            st.image(st.session_state['image_result'],st.session_state['process_type'])
        if st.session_state['process_type'] == 'gray histogram':
            st.pyplot(st.session_state['fig'])
        if st.session_state['process_type'] == 'color histogram':
            st.pyplot(st.session_state['color_fig'])
        if st.session_state['process_type'] == 'Blurring':
            st.image(st.session_state['blur_image'],st.session_state['process_type'])
        if st.session_state['process_type'] == 'edges of image':
            st.image(st.session_state['edge_image_canny'],'canny', clamp=True)
            st.image(st.session_state['edge_image_custom_sobel'],'custom_sobel')
        if st.session_state['process_type'] == 'DoG':
            st.image(st.session_state['difference_of_gaussian'],'difference of gaussian')
        if st.session_state['process_type'] == 'subtract two images':
            st.image(st.session_state['subtracted_image'],'subtracted')
        if st.session_state['process_type'] == 'sum of two images':
            st.image(st.session_state['sum_image'],'sum of two images')










# my_instance = st.session_state.my_instance

# if "step1" not in st.session_state:
#     st.session_state.step1=None

# if "uploaded_file" not in st.session_state:
#     st.session_state.uploaded_file= None
# if "form2_submit" not in st.session_state:
#     st.session_state.form2_submit=None
# form2_submit = st.session_state.form2_submit

# if "gen_edge_button" not in st.session_state:
#     st.session_state.gen_edge_button = False




# with st.sidebar:
#     st.header("denoise and Edge Filter Menu")


#     st.session_state.uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
#     st.session_state.denoise = st.radio("what denoise filter do you want to apply:",("Gaussian Filter","Median Filter","Mean Filter","Bilateral Filter"))
    
    
    
#     if st.session_state.uploaded_file and st.session_state.denoise is not None:
#         if st.button("submit desoise"):
#             if st.session_state.denoise == "Gaussian Filter":

#                 step1_result = my_instance.apply_gaussian_filter(image=st.session_state.uploaded_file,fsize=(6,6),sigma=5)

#             elif st.session_state.denoise == "Median Filter":
#                 step1_result = my_instance.median_blur(st.session_state.uploaded_file)
#             elif st.session_state.denoise == "Mean Filter":
#                 step1_result = my_instance.mean_filter(st.session_state.uploaded_file)
#             elif st.session_state.denoise == "Bilateral Filter":
#                 step1_result = my_instance.bilateral_filter(st.session_state.uploaded_file)
#             st.session_state.step1_result = step1_result
#             st.session_state.step1 = True
    
#     if st.session_state.uploaded_file and st.session_state.denoise and st.session_state.step1 is not None:
#         st.write("edge form")

#         st.session_state.edge = st.radio("what Edge filter do you want to apply:",("custom sobel","open_cv default sobel","Canny","otsu"))
#         st.session_state.gen_edge_button = st.button("Gen Edge")
        



# col1, col2, col3 = st.columns(3) 
# with col1: 
#     if st.session_state.uploaded_file is not None:
#         # Display the uploaded image
#         st.image(st.session_state.uploaded_file, caption='Uploaded Image', use_column_width=True)


# with col2:
#     if "step1_result" in st.session_state:
#         st.image(st.session_state.step1_result, caption='denoise image', use_column_width=True)
        


# #ToDo 
# with col3:
#     if st.session_state.gen_edge_button and st.session_state.step1_result is not None:
#         if st.session_state.edge=="custom sobel":
#             edge_image = my_instance.custum_compute_sobel(st.session_state.step1_result)
#         if st.session_state.edge=="open_cv default sobel":
#             edge_image = my_instance.cv2_compute_sobel(st.session_state.step1_result)
#         if st.session_state.edge=="Canny":
#             # edge_image = my_instance.(st.session_state.step1_result)
#             # st.image(edge_image, caption='edge image', use_column_width=True)
#             edge_image = my_instance.cv2_compute_sobel(st.session_state.step1_result)
#         if st.session_state.edge=="otsu":
#             edge_image = my_instance.otsu_thresholding(st.session_state.step1_result)
#             # edge_image.shape
#             len(edge_image)
#             # edge_image
#         st.image(edge_image, caption='edge image', use_column_width=True)