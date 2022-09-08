import pydicom as dicom
import cv2
from PIL import ImageTk, Image
import numpy as np

def what_image(self, path):
    image_format = Image.open(path)
    if image_format.format == 'JPEG':
        def read_jpg_file(image_format):
            img = cv2.imread(image_format)
            img_array = np.asarray(img)
            img2show = Image.fromarray(img_array) 
            img2 = img_array.astype(float) 
            img2 = (np.maximum(img2,0) / img2.max()) * 255.0
            img2 = np.uint8(img2)
            return img2, img2show 
    else:
        def read_dicom_file(image_format):    
            img = dicom.read_file(image_format)    
            img_array = img.pixel_array
            img2show = Image.fromarray(img_array)  
            img2 = img_array.astype(float) 
            img2 = (np.maximum(img2,0) / img2.max()) * 255.0
            img2 = np.uint8(img2)
            img_RGB = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)        
            return img_RGB, img2show




