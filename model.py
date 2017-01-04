from imagedataset import ImgDataSet
import numpy as np
import glob
import cv2
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Convolution2D, MaxPooling2D

# defining global variables
# image sizes after pre-processing  
img_rows = 48
img_cols = 96
img_ch= 1  
# directory in which data is saved
img_dir = 'C:\\Users\\ali.khalili\\Desktop\\Car-ND\\Car-ND-Behavioral-Cloning-P3\\data\\'    


def data_generator(num_images_to_load, batch_size):
  
  # at least batch_size number of images should be loaded into memory each time  
  assert num_images_to_load > batch_size
  
  # image sizes after pre-processing  
  global img_rows
  global img_cols
  global img_dir
  
  # setting up variables and dataframes
  file_names = np.asarray([f_name for f_name in glob.glob(img_dir+'IMG\\'+'*.jpg')])
  total_images = len(file_names)
  csv_data = pd.read_csv(img_dir+'driving_log.csv')
  
  # removing starting spaces
  csv_data['right'] = csv_data['right'].str.lstrip()
  csv_data['left'] = csv_data['left'].str.lstrip()
  csv_data['center'] = csv_data['center'].str.lstrip()
  
  while 1:     
    # Shuffling the images before loading
    perm = np.arange(len(file_names))
    np.random.shuffle(perm)
    file_names = file_names[perm]    
    
    # loading all images into memory and passing on to the optimizer in batches
    start_index = 0
    while start_index < total_images-1:
      # loading images and labels data into memory  
      loaded_images = []
      labels = []
      num_acutally_loaded = min(num_images_to_load,total_images-start_index)
      for i in range(num_acutally_loaded): 
        loaded_images.append(cv2.imread(file_names[start_index+i]))
        file_key = 'IMG/' + (file_names[start_index+i]).replace(img_dir+'IMG\\','')
        cam_type = file_key[file_key.index('/')+1:file_key.index('_')]
        if cam_type == 'center':
          steering_offset = 0
        elif cam_type == 'left':
          steering_offset = 0.25
        else:
          steering_offset = -0.25
        labels.append(csv_data[csv_data[cam_type]==file_key]['steering'].values[0]+steering_offset)
      
      # changing loaded images to ndarray
      loaded_images = np.asarray(loaded_images)
      labels = np.asarray(labels)
      start_index += num_acutally_loaded      
      
      # creating image dataset and pre-processing the images
      img_data_set = ImgDataSet(loaded_images, labels, scaled_dim=(img_cols,img_rows))
      img_data_set.pre_process()
      
      # passing on batches of data
      num_batches = num_acutally_loaded // batch_size
      for i in range(num_batches):
        yield img_data_set.next_batch(batch_size)


def get_model():
  
  # image sizes after pre-processing  
  global img_rows
  global img_cols
  global img_ch  
  
  model = Sequential()
  
  # Convolution 1
  kernel_size = (5,5)
  nb_filters = 24
  model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='valid',input_shape=(img_rows, img_cols, img_ch)))
  # Pooling
  pool_size = (2,2)
  model.add(MaxPooling2D(pool_size=pool_size))
  # Dropout
  keep_prob = 0.5
  model.add(Dropout(keep_prob))
  
  # Convolution 2
  kernel_size = (5,5)
  nb_filters = 36
  model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='valid'))
  # Pooling
  pool_size = (2,2)
  model.add(MaxPooling2D(pool_size=pool_size))
  # Activation
  model.add(Activation('relu'))
  
  # Convolution 3
  kernel_size = (4,4)
  nb_filters = 48
  model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='valid'))
  # Pooling
  pool_size = (2,2)
  model.add(MaxPooling2D(pool_size=pool_size))
  # Activation
  model.add(Activation('relu'))
  
  # Convolution 4
  kernel_size = (3,3)
  nb_filters = 64
  model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='valid'))
  
  # flatten
  model.add(Flatten())
  
  # fully connected 1
  model.add(Dense(100))
  
  # fully connected 2
  model.add(Dense(35))
  
  # fully connected 3
  model.add(Dense(1))
  
  # compiling the model
  model.compile(optimizer="adam", loss="mse")
  
  return model


def main():
  model = get_model()
  model.fit_generator(data_generator(1000,64),samples_per_epoch=24108,nb_epoch=2)


if __name__ == '__main__':
  main()
    