#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[bar_frequency]: ./resource/bar_training.png "Frequency_dataset"
[original_image]: ./resource/original_image.png "Original data"
[augmented_images]: ./resource/augmentation.png  "augmented data"
[test_images]: ./resource/test_images.png  "test data"
[top5]:./resource/classification_result.png  "classification_result"
[top5_2]:./resource/classification_result2.png  "classification_result2"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/sir-siemens/carND-project2/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ? 34799
* The size of test set is ?  12630
* The shape of a traffic sign image is ? 32x32
* The number of unique classes/labels in the data set is ? 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is implemented in the function of statistics_visualization()

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distributed across different class
![123][bar_frequency]


###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

I only apply the normalization of the image data. The code for this step is implemented 

def preprocessing(X_train)

I have also tried to convert image to gray scale, however the validation accuracy is not high. So I decide to use 3 channels to the network and augment my dataset in color scale. The intuition is the color feature should have a contribution to the classification result. 

 
####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the first code cell of the IPython notebook.  

My final training set had 4000*43 = 172000 number of images. My validation set and test set are loaded from the provided data

The code cell 'Balencing training set' of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because the training data is unbalanced. To add more data to the the data set, I apply perspective transformation and adjusting brightness and contrast to balance the training data. They are implemented in the following functions, because it add additional variation to the training data and are similar to the training data.  
* random_perspective_transform(image)           (used in final version))
* contrast_brightness_rgb(image, alpha , beta)  (used in final version))
* histogram_equalization(image)                 (not used in final version)
* rgb2gray(image)                               (not used in final version) 
* contrast_brightness_gray(image, alpha , beta) (not used in final version)


Here is an example of a traffic sign image before and after augmentation.

original image

![alt text][original_image]

augmentated image

![alt text][augmented_images]



####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the section of Model Architecture (Modified Lenet + dropout)

My final model consisted of is basically the same as Lenet5:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x6 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3	    |  1x1 stride, valid padding, outputs 12x12x16		|
| RELU					|												|
| Convolution 3x3	    |  1x1 stride, valid padding, outputs 10x10x16		|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		|  400x120    									|
| RELU		|    									|
| Dropout		| keep_prob = 0.5    									|
| Fully connected		|  120x84    									|
| RELU		|    									|
| Dropout		| keep_prob = 0.5    									|
| Fully connected		|  84x10    									|
 

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the section of Train, Validate and Test the Model

To train the model, I use a pre-train stage and a fine tuning stage. In the pre-train stage, I train about 10 epoch, learning rate 0.001 and L2 regularzation. And in the fine tuning stage, I lower the learning rate to 0.0005 and train for 100 epoch. For both stage I chooset the batch size to be 128

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in training pipeline

My final model results were:
* training set accuracy of 0.961
* validation set accuracy of 0.988
* test set accuracy of 0.975

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? 
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

I first choose standard LeNet, because I have no idea what else may be fit for this problem. For the standard LeNet, there are no dropout layers, so I added in the final version and it proved to be a big gain in validation accuracy. One problem that I encountered at the beginning is I get a training accuracy of 100%, however the network fails to generalize well on validation dataset. It seems the network overfits the training set, so I tried both augmentating the training data and using drop out layers, which results in a higher validation accuracy and a lower training accuracy. For tuning, I first use only one dropout layer, then I decide to put dropout layers in all the fully connected layer. In addition, increasing the training EPOCH is also important to catch the model with the highest validation accuracy. I finally got 96.1% test accuracy. Then I modify the architecture try to see whether make the model deep may help, so I follow the design pattern of a convnet introduced in the CS231n: Convolutional Neural Networks for Visual Recognition and add 4 convnet followed by fully connected. And I get a performance gain of 1.4 % in test accuracy.

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? 

Training accuracy is lower than validation accuracy may indicate my model does not overfit the data. In addition, I'm quite satisfy with the test accurcy. 
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][test_images]  

The last image (speed limit from eletronic LED display on highway) might be difficult to classify because it is not provided in the training set. 

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the section of 'Predict the Sign Type for Each Image'


Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Wild animals crossing   		| Wild animals crossing   									| 
| No entry 			| No entry 										|
| Slippery road					| Slippery road											|
| roundabout mandatory	      		|roundabout mandatory		 				|
| speed limit 60			| speed limit 30     							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. (for both LeNet+Dropout and my modified version). 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the last cell of the Ipython notebook.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| ~0.99         			| Wild animals crossing   									| 
| ~0.99    				|  No entry 							 										|
| ~0.99					| Slippery road											|
| ~0.99	      			| roundabout mandatory				 				|
| ~0.20   			    |  speed limit 60	    							|

Result of Lenet+Dropout
![alt text][top5]

Result of ModifiedLenet
![alt text][top5_2]



###Final Notes:
I invest significantly more time in this project than in P1. Thanks to the available deep learning framework, a novice can learn this technique in quite a short time and train their models. However, the upside of this technique is also the downside of this technique. It automatically learns the features which are relevant to the problem means at the same time, we can not influence them mannually. The only information that we can use to tune the model is the validation accuracy, which makes tuning the model actually a challenging task. 
