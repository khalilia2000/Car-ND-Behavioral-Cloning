# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 13:03:29 2017

@author: Ali Khalili
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import gridspec


class ImgDataSet(object):

  def __init__(self,images,labels,norm_max_min=0.5,scaled_dim=(96,48)):
               
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
    
    # The following parameters will be used for retrieving data batches    
    self._epochs_completed = 0
    self._index_in_epoch = 0

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
    """
    Adds horizontally flipped images to the dataset
    """
    self._images = np.append(self._images, np.asarray([cv2.flip(image,1) for image in self._images]), axis=0)
    self._labels = np.append(self._labels, np.asarray([-steer_angle for steer_angle in self._labels]), axis=0)
    self._num_examples = self._images.shape[0]
    pass  


  def color_to_gray(self):
    """
    Changes the color to grayscale (i.e. turns images to only one channel)
    """
    self._images = np.asarray([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in self._images])
    pass

  
  def equalize_hist(self):
    """
    Equalizes the histogram
    """
    self._images = np.asarray([cv2.equalizeHist(img) for img in self._images])
    pass


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

  
  def normalize(self):
    """
    Normalizes the piexel values of the images
    """
    self._images = self._images.astype('float32')
    self._images -= 127.5
    self._images /= 127.5
    self._images /= self._normalization_factor
    pass
  
  
  def pre_process(self, add_flipped=True):
    """
    Executes the pipeline for preprocessing the images
    """
    if add_flipped:    
      self.add_flipped()
    self.color_to_gray()
    self.crop_top()
    self.resize()
    self.equalize_hist()
    self.normalize()
    # changing shape of the images to contain 1 channel
    self._images = self._images.reshape(-1,self._images.shape[1],self._images.shape[2],1)
    
    pass
  
  
  def next_batch(self, batch_size):
    """
    Return the next "batch_size" examples from this data set.
    """
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Start next epoch 
      # No need for suffling the data as generator shuffles the files before loading
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]
 

  def plot_image(self, ax_list, grid_fig, grid_index, image_index, show_label=True, cmap=None):
    """
    plots one image in the grid space and passess the appended axis list
    """
    ax_list.append(plt.subplot(grid_fig[grid_index]))
    
    ax_list[-1].imshow(self._images[image_index], cmap=cmap)
    ax_list[-1].axis('off')
    ax_list[-1].set_aspect('equal')
    y_lim = ax_list[-1].get_ylim()
    if show_label:
      ax_list[-1].text(0,int(-1*y_lim[0]*0.1),'Steering Angle = {}'.format(self._labels[image_index]),color='r')
      ax_list[-1].text(0,int(-1*y_lim[0]*0.3),'Image Index = {}'.format(image_index),color='b')

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
        
      