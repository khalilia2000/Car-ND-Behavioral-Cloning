# Car-ND-Behavioral-Cloning

Third Project of the Car Nano-degree from Udacity. The following approach was taken during the implementation of this project:

#### 1- Literature Review:  
A review of the Past Work and Existing Literature (i.e. including posts by other students) was done:
I was especially inspired by the following links:  
   * https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.1j27mpyik  
   * https://medium.com/@tantony/training-a-neural-network-in-real-time-to-control-a-self-driving-car-9ee5654978b7#.ekb6aognp  
   * https://carnd-forums.udacity.com/pages/viewpage.action?pageId=32113760  
   * http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf  
   * https://github.com/diyjac/AgileTrainer  
   * https://github.com/commaai/research/blob/master/train_steering_model.py  
   * https://carnd-forums.udacity.com/questions/26214464/answers/26215455  

#### 2- Trainind Data and Pre-Processing:  
The database provided by Udacity was used for initial training and an ImgDataSet class was created for pre-processing the data. The preprocessing pipeline included the following steps:
   * adding horizontally flipped images with reversed steering angles to the database  
   * adding images from left and right cameras and adjusting steering angles by 0.25 in each case  
   * changing the images to grayscale  
   * croping the top of images to avoid training the model on anything above the horizon (i.e. the sky) and saving memory  
   * resizing the images to 96x48 pixels to reduce memory consumption  
   * equalizing the histograms of all images to have a more central distribution  
   * normalizing the pixel values to +/-0.5  
   Sample images from the training dataset before and after the pre-processing are included below:  
     
   ![alt text](https://github.com/khalilia2000/Car-ND-Behavioral-Cloning/blob/master/example_data_1.JPG "Training Data Before and After Pre-Processing")

#### 3- Data Generator:  
a data generator was used to feed data into fit_generator and to keep the memory consumption managable. The data generator reads images in blocks from the disk and passes them in batch_size to the fit_generator function.

#### 4- Model Architecure:  
I started with the NVIDIA's end to end architecture and incrementally made changes to the network. Each incremental network change consisted of training 3 to 5 epochs on the database, and evaluating the results based on the MSE for the validation set. The final network architecture that I settled for is the following (the serial Convolution2D and Relu activation units should result in non-linearity of the process):  

	| Layer (type)                     | Output Shape         | Param #    | Connected to                 |     
	|:-------------------------------- |:-------------------- |:---------- |:---------------------------- |     
	| convolution2d_1 (Convolution2D)  | (None, 44, 92, 36)   | 936        | convolution2d_input_21[0][0] |     
	| maxpooling2d_1 (MaxPooling2D)    | (None, 22, 46, 36)   | 0          | convolution2d_1[0][0]        |     
	| dropout_1 (Dropout)              | (None, 22, 46, 36)   | 0          | maxpooling2d_1[0][0]         |     
	| convolution2d_2 (Convolution2D)  | (None, 18, 42, 36)   | 32436      | dropout_1[0][0]              |     
	| maxpooling2d_2 (MaxPooling2D)    | (None, 9, 21, 36)    | 0          | convolution2d_2[0][0]        |     
	| activation_1 (Activation - Relu) | (None, 9, 21, 36)    | 0          | maxpooling2d_2[0][0]         |     
	| convolution2d_3 (Convolution2D)  | (None, 6, 18, 48)    | 27696      | activation_1[0][0]           |     
	| maxpooling2d_3 (MaxPooling2D)    | (None, 3, 9, 48)     | 0          | convolution2d_3[0][0]        |     
	| activation_2 (Activation - Relu) | (None, 3, 9, 48)     | 0          | maxpooling2d_3[0][0]         |     
	| convolution2d_4 (Convolution2D)  | (None, 1, 7, 64)     | 27712      | activation_2[0][0]           |     
	| flatten_1 (Flatten)              | (None, 448)          | 0          | convolution2d_4[0][0]        |     
	| dense_1 (Dense)                  | (None, 100)          | 44900      | flatten_1[0][0]              |     
	| dense_2 (Dense)                  | (None, 50)           | 5050       | dense_1[0][0]                |     
	| dense_3 (Dense)                  | (None, 16)           | 816        | dense_2[0][0]                |     
	| dense_4 (Dense)                  | (None, 1)            | 17         | dense_3[0][0]                |     

	Total params: 139,563
	Trainable params: 139,563
	Non-trainable params: 0
	

#### 5- Fine-Tuning the Model:  
Once settled on the model architecture, I trained the model using the database provided by Udacity for approximately 10 epochs (each epoch took about 5 minutes on my machine). Once done, the car could drive in autonomous condition for most parts on track 1, except in few isolated locations where it would go off-track. I then generated additional datasets by using training mode of the simulator and fine-tuned my network by training a few epochs (i.e. typically less than 4) on these additional datasets. After a few fine-tuning rounds, the car could drive autonomously on track 1. Each of these datasets pertained to the locations where the car would go off-track. I generated the datasets by parking the car close to the curb, and brining it back to the centre. I only used images in which streeing angle was not zero (i.e. deleted those that steering angle was zero). Examples of additional training data are included below:  
  
  ![alt text](https://github.com/khalilia2000/Car-ND-Behavioral-Cloning/blob/master/example_data_2.JPG "Training Data Before and After Pre-Processing")

#### 6- Choice of hyper parameters:   
   * learning rate: since I was using Adam optimizer, I used the default learning rate of 0.001 for initial training, whcih worked OK. However, for fine-tuning of the network weights, I found out that using smaller initial learning rates work better. I ended up using learning rates of between 0.00002 and 0.0002. Note that Adam optimizer adjusts the learning rate during training to some extent.  
   * batch size: I played with batch sizes and tried batch sizes of 23, 64, and 128. I found that a batch size of 64 was giving the most stable resutls with fater convergance.
    
#### 7- Avoiding over-fitting:  
To avoid overfitting I did the following:
   * Dropout Layer: I added a dropout layer with keep_probability of 50% after the first convolution and pooling layer. I also tried adding two dropout layers and it did not result in a significant improvement.
   * Splitting Data int Training and Validation Datasets: I did split the data into validation and training datasets with approximtely 15% data set aside as validation.
   * No test dataset was set aside, because the ultimate test was considered to be the driving of the car autonomously on the first track.
	
## Reflection on the results
The following oversvatinos were noted from the results and during training/testing the models:  
   * Training on the original dataset provided a more smooth driving experience as opposed to training on the datasets that I created myself possibly due to the fact that I was using a keyboard for training, which resulted in overcompensating in some cases. Following the same line, in some areas on the track the car appears to behave erratically and over-compensating for steering angle. This can be attributed to the fact that additional training data I generated was with keyboard.
   * After the first successful model was trained with data from track 1, I tried the model on track 2, and found out that it goes off-track is a few locations. I had to fine-tune the model even more using sharp corners as training data to overcome that and make the car drive successfully on track 2 too.
   * I had to increase the throttle to 0.3 in drive.py in order for the car to be able to go up the hill on track 2 (as opposed to throttle of 0.2 being sufficient for track 1)
