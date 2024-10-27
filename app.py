import streamlit as st
import cv2
import numpy as np
from scipy import ndimage

st.title("Image Analysis with Gabor Filters and Fractalius Effect")

#upload an Image
uplaoded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uplaoded_file is not None:
    #read image with OpenCV
    image = cv2.imdecode(np.frombuffer(uplaoded_file.read(), np.uint8), 1)
    st.image(image, channels="BGR", caption="Original Image")



def apply_gabor_filter(image, frequency, theta):

#create  a gabor kernel
    kernel = cv2.getGaborKernel((21, 21), 8.0, theta, frequency, 0.5, 0, ktype=cv2.CV_32F)
    filtered_image = cv2.filter2D(image, cv2.CV_8UC3, kernel)
    return filtered_image

#add sliders for gabor filter param
if st.sidebar.checkbox("Apply Gabor Filter"):
    frequency = st.sidebar.slider("Frequency (Wavelength)", 1, 20, 5) 
    theta = st.sidebar.slider("Orientation (Angle)", 0, 180, 45)
    gabor_filtered_image = apply_gabor_filter(image, frequency, np.deg2rad(theta)) 
    st.image(gabor_filtered_image, caption="Gabor Filter Applied") 


#implementing Fractalius Effect
def apply_fractalius_effect(image):
    #convert image to grayscale and apply edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=5)
    #enhance edges and overlay
    edges_colored = cv2.applyColorMap(edges, cv2.COLORMAP_JET)
    fractalius_image = cv2.addWeighted(image, 0.7, edges_colored, 0.3, 0)
    return fractalius_image

#add checkbox for fractalius effect
if st.sidebar.checkbox("Apply Fractalius Effect"):
    fractalius_effect = apply_fractalius_effect(image)
    st.image(fractalius_effect, caption="Fractalius Effect Applied")