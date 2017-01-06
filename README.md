# Car-ND-Behavioral-Cloning

Third Project of the Car Nano-degree from Udacity. The following approach was taken during the implementation of this project:

1- I did a review of Past Work and Existing Literature (i.e. including posts by other students):
I was especially inspired by the following links:  
   * https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.1j27mpyik  
   * https://medium.com/@tantony/training-a-neural-network-in-real-time-to-control-a-self-driving-car-9ee5654978b7#.ekb6aognp  
   * https://carnd-forums.udacity.com/pages/viewpage.action?pageId=32113760  
   * http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf  
   * https://github.com/diyjac/AgileTrainer  
   * https://github.com/commaai/research/blob/master/train_steering_model.py  
   * https://carnd-forums.udacity.com/questions/26214464/answers/26215455  

 2- I used the database provided by Udacity and created a ImgDataSet class for pre-processing the data. The preprocessing pipeline included the following steps:
   * adding horizontally flipped images with reversed steering angles to the database  
   * adding images from left and right cameras and adjusting steering angles by 0.25 in each case  
   * changing the images to grayscale  
   * croping the top of images to avoid training the model on the sky and saving memory  
   * resizing the images to 96x48 pixels to reduce memory consumption  
   * equalizing the histograms of all images to have a more central distribution  
   * normalizing the pixel values to +/-0.5  

 3- I used a data generator to feed data into fit_generator and keep the memory consumption managable. The data generator reads images in blocks from the disk and passes them in batch_size specified to the fit_generator.

 4- I started with the NVIDIA's end to end architecture and incrementally made changes to the network. Each incremental network change consisted of training 3 to 5 epochs on the database, and evaluating the results. The final network architecture that I settled for is the following:  
    Layer (type)                     Output Shape          Param #     Connected to                     
	====================================================================================================
	convolution2d_1 (Convolution2D)  (None, 44, 92, 36)    936         convolution2d_input_21[0][0]     
	____________________________________________________________________________________________________
	maxpooling2d_1 (MaxPooling2D)    (None, 22, 46, 36)    0           convolution2d_1[0][0]            
	____________________________________________________________________________________________________
	dropout_1 (Dropout)              (None, 22, 46, 36)    0           maxpooling2d_1[0][0]             
	____________________________________________________________________________________________________
	convolution2d_2 (Convolution2D)  (None, 18, 42, 36)    32436       dropout_1[0][0]                  
	____________________________________________________________________________________________________
	maxpooling2d_2 (MaxPooling2D)    (None, 9, 21, 36)     0           convolution2d_2[0][0]            
	____________________________________________________________________________________________________
	activation_1 (Activation)        (None, 9, 21, 36)     0           maxpooling2d_2[0][0]             
	____________________________________________________________________________________________________
	convolution2d_3 (Convolution2D)  (None, 6, 18, 48)     27696       activation_1[0][0]               
	____________________________________________________________________________________________________
	maxpooling2d_3 (MaxPooling2D)    (None, 3, 9, 48)      0           convolution2d_3[0][0]            
	____________________________________________________________________________________________________
	activation_2 (Activation)        (None, 3, 9, 48)      0           maxpooling2d_3[0][0]             
	____________________________________________________________________________________________________
	convolution2d_4 (Convolution2D)  (None, 1, 7, 64)      27712       activation_2[0][0]               
	____________________________________________________________________________________________________
	flatten_1 (Flatten)              (None, 448)           0           convolution2d_4[0][0]            
	____________________________________________________________________________________________________
	dense_1 (Dense)                  (None, 100)           44900       flatten_1[0][0]                  
	____________________________________________________________________________________________________
	dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]                    
	____________________________________________________________________________________________________
	dense_3 (Dense)                  (None, 16)            816         dense_2[0][0]                    
	____________________________________________________________________________________________________
	dense_4 (Dense)                  (None, 1)             17          dense_3[0][0]                    
	====================================================================================================
	Total params: 139,563
	Trainable params: 139,563
	Non-trainable params: 0
	____________________________________________________________________________________________________


 5- fine-tuning the model: once settled on the model architecture, I trained the model using the database for approximately 10 epochs. Once done, the car could drive in autonomous condition on track 1, excep in few isolated locations where it would to off-road. I then generated additional datasets by using training mode of the simulator and fine-tuned my network by training a few epochs (i.e. typically less than 3) on these additional datasets. After a few fine-tuning rounds, the car could drive autonomously on track 1.

 6- Choice of hyper parameters:
    * learning rate: since I was using Adam optimizer, I used the default learning rate of 0.001 for initial training. Subsequently for fine-tuning the network weights, I used a much smaller initial learning rate of 0.0002 to avoid over-fitting in those areas. Note that Adam optimizer adjusts the learning rate during training.