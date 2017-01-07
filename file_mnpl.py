# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 11:53:27 2017

@author: ali.khalili
"""

import glob
import numpy as np
import pandas as pd
import os


work_dir =  'C:\\Users\\ali.khalili\\Desktop\\Car-ND\\Car-ND-Behavioral-Cloning-P3\\'
img_dir = work_dir + 'added_data5\\' 

def main():
  
  filenames = np.asarray([f_name for f_name in glob.glob(img_dir+'IMG\\'+'*.jpg')])
  total_images = len(filenames)
  csv_data = pd.read_csv(img_dir+'driving_log.csv')
  
  # removing starting spaces
  csv_data['right'] = csv_data['right'].str.lstrip()
  csv_data['left'] = csv_data['left'].str.lstrip()
  csv_data['center'] = csv_data['center'].str.lstrip()
  
  counter = 0
  for i in range(total_images):
    file_key = 'IMG/' + (filenames[i]).replace(img_dir+'IMG\\','')
    cam_type = file_key[file_key.index('/')+1:file_key.index('_')]
    if file_key not in csv_data[cam_type].values:
      os.remove(filenames[i])
      counter += 1
  print("A total of {} files were removed: ".format(counter))
  pass

  



if __name__ == '__main__':
  main()