# Image-Sentiment-Classification

Classifying the images based on the facial expressions/emotions using a Convolution Neural Network (CNN) and Transfer learning approach via VGG-16 network.


##Data
Each record represents a single image having two attributes/columns: 'feature', 'label'
1. 'feature' column consists of 48*48=2034 pixel values for a particular image.

2. 'label' column indicates the type of emotion (happy, angry, sad, etc.) depicted by the image.


##Data Preprocessing
Data is pre-processed and transformed using Data Augmentation techniques (Rotate shifting, flipping, cropping, image distortions, etc.) using the ImageDataGenerator module.


##Modeling - 2 models were built
1. CNN model was built which consists of 5 main convolutional blocks (each block with 2Convolution layers, 1 Max-Pooling layer followed by Batch Normalization and Drop out layers), 2 Fully Connected layers, and Output Dense layer.
2. Transfer Learning: Pre-trained VGG-16 model (a 16-layer network built on the ImageNet database) was used as well, where a new fully-connected layer is added at the end of the network, initialized with random weights, and trained the FCN and the weights from the pre-trained network are frozen (Convolutional blocks) in order to use the same weights which were learned through pre-training on the ImageNet.


##Results: The CNN model performed better which resulted in higher accuracy than the pre-trained VGG-16 model.


##Interpretable ML - Used Saliency maps to analyze and plot the gradient of the predicted outcome from the model with respect to the input.
