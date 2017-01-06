# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 14:48:08 2017

@author: Ali.Khalili
"""


from imagedataset import ImgDataSet
from model import get_filenames_and_labels
import numpy as np
import cv2


# defining global variables
# image sizes after pre-processing  
img_rows = 48
img_cols = 96
img_ch= 1  
img_norm = 0.5 # max/min of normalized pixel values
# directory in which data is saved
work_dir =  'C:\\Users\\ali.khalili\\Desktop\\Car-ND\\Car-ND-Behavioral-Cloning-P3\\'
img_dir = work_dir + 'added_data\\' 
# maximum number of images to read from disk into memory
max_mem = 5


def load_and_print_images():
  
  global img_rows
  global img_cols
  global img_ch
  global img_norm
  global work_dir
  global img_dir
  global max_mem
  
  # splitting data into training and validatoin sets
  X_trn, y_trn, X_val, y_val = get_filenames_and_labels()
  
  perm = np.arange(len(X_trn))
  np.random.shuffle(perm)
  filenames = X_trn[perm]    
  labels = y_trn[perm]
  
  loaded_images = []
  num_loaded = max_mem
  for i in range(num_loaded): 
    loaded_images.append(cv2.imread(filenames[i]))
  loaded_images = np.asarray(loaded_images)
  
  sliced_labels = labels[:num_loaded]
  img_data_set = ImgDataSet(loaded_images, sliced_labels, norm_max_min=img_norm, scaled_dim=(img_cols,img_rows))
  img_data_set.plot_random_grid(1,5)

  img_data_set.pre_process()
  img_data_set.plot_random_grid(1,5,cmap='gray')
  
  return img_data_set

  

def main():
  pass
  



if __name__ == '__main__':
  main()