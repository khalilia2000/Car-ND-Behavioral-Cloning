from imagedataset import ImgDataSet
import numpy as np
import glob
import cv2
import pandas as pd
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import json

# defining global variables
# image sizes after pre-processing  
img_rows = 48
img_cols = 96
img_ch= 1  
img_norm = 0.5 # max/min of normalized pixel values
# directory in which data is saved
work_dir =  'C:\\Users\\ali.khalili\\Desktop\\Car-ND\\Car-ND-Behavioral-Cloning-P3\\'
img_dir = work_dir + 'added_data2\\' 
# maximum number of images to read from disk into memory
max_mem = 5120


def get_filenames_and_labels(test_ratio=0.15):
  """
  helper function for constructing arrays of labels and image filenames and
  splitting them into training and validation sets
  """
  
  global img_dir
  
  filenames = np.asarray([f_name for f_name in glob.glob(img_dir+'IMG\\'+'*.jpg')])
  total_images = len(filenames)
  csv_data = pd.read_csv(img_dir+'driving_log.csv')
  
  # removing starting spaces
  csv_data['right'] = csv_data['right'].str.lstrip()
  csv_data['left'] = csv_data['left'].str.lstrip()
  csv_data['center'] = csv_data['center'].str.lstrip()

  # setting labels corresponding to the image filenames
  labels = []
  for i in range(total_images):
    file_key = 'IMG/' + (filenames[i]).replace(img_dir+'IMG\\','')
    cam_type = file_key[file_key.index('/')+1:file_key.index('_')]
    if cam_type == 'center':
      steering_offset = 0
    elif cam_type == 'left':
      steering_offset = 0.25
    else:
      steering_offset = -0.25
    labels.append(csv_data[csv_data[cam_type]==file_key]['steering'].values[0]+steering_offset)
    
  # converting labels to ndarray  
  labels = np.asarray(labels)
  
  # splitting the data into training and validation sets  
  X_train, X_val, y_train, y_val = train_test_split(filenames,labels,test_size=test_ratio,stratify=[1 for x in labels])
  
  return X_train, y_train, X_val, y_val
  
  

def data_generator(num_images_to_load, batch_size, training_filenames, training_labels):
  """
  data generator for training data
  num_iamges_to_load: is the number images that will be loaded into memory each time the files are read from disk
  batch_size: is the batch_size of training data that is yielded in the generator
  training_filenames: is the ndarray of filenames containing the training data
  training_labels: is the ndarray of labels corresponding to the filenames
  """
  # at least batch_size number of images should be loaded into memory each time  
  assert num_images_to_load >= batch_size
  # number of images and labels should be the same
  assert training_filenames.shape[0] == training_labels.shape[0]
  
  # image sizes after pre-processing  
  global img_rows
  global img_cols
  global img_dir
  global img_norm
  
  # length of the training dataset
  total_trn_images = len(training_filenames)
  
  while 1:     
    # Shuffling the images before loading
    perm = np.arange(len(training_filenames))
    np.random.shuffle(perm)
    filenames = training_filenames[perm]    
    labels = training_labels[perm]
    
    # loading images into memory in batches and passing on to the optimizer
    start_index = 0
    while start_index < total_trn_images-1:
      
      # loading image data into memory  
      loaded_images = []
      num_loaded = min(num_images_to_load,total_trn_images-start_index)
      for i in range(num_loaded): 
        loaded_images.append(cv2.imread(filenames[start_index+i]))
      loaded_images = np.asarray(loaded_images)
            
      # creating image dataset and pre-processing the images
      sliced_labels = labels[start_index:start_index+num_loaded]
      img_data_set = ImgDataSet(loaded_images, sliced_labels, norm_max_min=img_norm, scaled_dim=(img_cols,img_rows))
      img_data_set.pre_process()
      
      # passing on batches of data
      num_batches = img_data_set.num_examples // batch_size
      for i in range(num_batches):
        yield img_data_set.next_batch(min(batch_size,num_loaded))
        
      # adjusting and moving forward the start_index
      start_index += num_loaded      


def get_model():
  
  # image sizes after pre-processing  
  global img_rows
  global img_cols
  global img_ch  
  
  model = Sequential()
  
  # Convolution 1
  kernel_size = (5,5)
  nb_filters = 36
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
  model.add(Dense(50))
  
  # fully connected 3
  model.add(Dense(16))
  
  # fully connected 4
  model.add(Dense(1))
  
  # compiling the model
  model.compile(optimizer="adam", loss="mse") #using default hyper parameters when creating new network
  
  return model

def load_model_and_train(model_file, X_trn, y_trn, X_val, y_val, epochs=10, b_size=64):
  """
  loads the model and weights that were previously saved, and
  continues training the model for the number of epochs specified.
  """
  
  global max_mem
  
  # loading the model
  with open(model_file, 'r') as jfile:
    json_str = json.loads(jfile.read())
    model = model_from_json(json_str)

  adam_opt = Adam(lr=0.0002) # reducing learning rate for trainin on additional data
  model.compile(optimizer=adam_opt, loss="mse")
  weights_file = model_file.replace('json', 'h5')
  model.load_weights(weights_file)
  
  # training the model
  num_trn_samples = ((len(X_trn)*2) // b_size) * b_size
  num_val_samples = ((len(X_val)*2) // b_size) * b_size
  model.fit_generator(data_generator(max_mem,b_size,X_trn,y_trn),
                      samples_per_epoch=num_trn_samples,nb_epoch=epochs, 
                      validation_data=data_generator(max_mem,b_size,X_val,y_val), 
                      nb_val_samples=num_val_samples)  
  
  return model
  

def build_model_and_train(X_trn, y_trn, X_val, y_val, epochs=10, b_size=64):
  """
  builds a model with random weights and trains the model
  """
  
  global max_mem
  
  # creating the model
  model = get_model()
  # training the model
  num_trn_samples = ((len(X_trn)*2) // b_size) * b_size
  num_val_samples = ((len(X_val)*2) // b_size) * b_size
  model.fit_generator(data_generator(max_mem,b_size,X_trn,y_trn),
                      samples_per_epoch=num_trn_samples,nb_epoch=epochs, 
                      validation_data=data_generator(max_mem,b_size,X_val,y_val), 
                      nb_val_samples=num_val_samples)
  return model
  
  

def save_model_and_weights(model):
  """
  saves the model structure and the weights
  """
  global work_dir
  # saving the model and its weights
  print()
  print('Saving model and weights...')
  model.save_weights(work_dir+'model.h5')
  with open(work_dir+'model.json','w') as outfile:
    json.dump(model.to_json(), outfile)
  print('Done.')
  pass

  
  
def main():
  
  # splitting data into training and validatoin sets
  test_ratio = 0.15 # ratio of validation images to total number of images
  X_trn, y_trn, X_val, y_val = get_filenames_and_labels(test_ratio)
  
  # loading model and training
  model_file = work_dir+'model.json'
  model = load_model_and_train(model_file, X_trn, y_trn, X_val, y_val, epochs=2)
  
  save_model_and_weights(model)
  



if __name__ == '__main__':
  main()
    