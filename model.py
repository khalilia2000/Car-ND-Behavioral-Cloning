# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 13:03:29 2017

@author: Ali
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import gridspec
import glob


class ImgDataSet(object):

  def __init__(self,images,labels,norm_max_min=0.005,scaled_dim=(100,50)):
               
    """
    Construct a DataSet of Images.
    """
        
    assert images.shape[0] == labels.shape[0], (
        'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
    
    self._num_examples = images.shape[0]
    self._images = images
    self._labels = labels
    self._size_factor = 3
    self._aspect_ratio = 1.0 * images.shape[1] / images.shape[2]
    self._normalization_factor = 1 / norm_max_min
    self._resize_dim = scaled_dim


  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples


  def add_flipped(self):
    self._images = np.append(self._images, np.asarray([cv2.flip(image,1) for image in self._images]), axis=0)
    self._labels = np.append(self._labels, np.asarray([-steer_angle for steer_angle in self._labels]), axis=0)
    self._num_examples = self._images.shape[0]
    pass  


  def color_to_gray(self):
    self._images = np.asarray([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in self._images])
    pass

  
  def equalize_hist(self):
    """
    Equalizes the histogram
    """
    self._images = np.asarray([cv2.equalizeHist(img) for img in self._images])
    pass

  
  def normalize(self):
    """
    Normalizes the piexel values of the images
    """
    self._images = self._images.astype('float32')
    self._images -= 127.5
    self._images /= 127.5
    self._images /= self._normalization_factor


  def crop_top(self):
    """
    Crops the top portion of the image
    """
    if len(self._images[0].shape) == 2:
      self._images = np.asarray([img[-100:,:] for img in self._images])
    else:
      self._images = np.asarray([img[-100:,:,:] for img in self._images])
    pass
  
  
  def resize(self):
    """
    Resizes the images
    """
    self._images = np.asarray([cv2.resize(img, self._resize_dim) for img in self._images])
    pass
    

  def plot_image(self, ax_list, grid_fig, grid_index, image_index, show_label=True, cmap=None):
    """
    plots one image in the grid space and passess the appended axis list
    """
    ax_list.append(plt.subplot(grid_fig[grid_index]))
    
    ax_list[-1].imshow(self._images[image_index], cmap=cmap)
    ax_list[-1].axis('off')
    ax_list[-1].set_aspect('equal')
    if show_label:
      ax_list[-1].text(0,-15,'Steering Angle = {}'.format(self._labels[image_index]),color='r')
      ax_list[-1].text(0,-45,'Image Index = {}'.format(image_index),color='b')

    return ax_list

  
  def plot_random_grid(self, n_rows, n_cols, show_labels=True, cmap=None):
    """
    plots a random grid of images to verify
    """
    
    g_fig = gridspec.GridSpec(n_rows,n_cols) 
    g_fig.update(wspace=0.5, hspace=0.75)
    
    # setting up the figure
    plt.figure(figsize=(n_cols*self._size_factor,n_rows*self._size_factor*self._aspect_ratio))
    selection = np.random.choice(self._num_examples, n_rows*n_cols, replace=False)
    
    ax_list = []
    for i in range(n_rows*n_cols):
      ax_list = self.plot_image(ax_list, g_fig, i, selection[i], cmap=cmap)
        
      

def main():
  img_dir = 'C:\\Udacity Courses\\Car-ND-Udacity\\P3 - Behavioral Cloning\\data\\data\\IMG\\'    
  image_list = []
  file_names = [f_name for f_name in glob.glob(img_dir+'*.jpg')]
  num_images_to_load = 10
  total_images = len(file_names)
  for i in range(num_images_to_load): 
    index = int(np.random.uniform()*total_images)
    im=cv2.imread(file_names[index])
    image_list.append(im)    
  
  image_list = np.asarray(image_list)  
  labels = np.asarray([1 for x in image_list])
  #print(image_list.shape)
  ids1 = ImgDataSet(image_list,labels,1)
  return ids1

if __name__ == '__main__':
  main()
    