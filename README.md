# FER-of-Emotion-with-CNN
Facial Expression Recognition of Emotion using Deep Learning Convolutional Neural Network

I used FER2013 as dataset, because I consider it a good starting to point to keep it simple in term of complexity of size and computationally of the dataset,
but also to make it hard to achieve a good accuracy, as FER2013 is an outdated dataset, unbalance and poor of details. It contains 35887 images of facial expression examples in grayscale, for unbalanced I mean e.g., the largest class, happy, containing more data 8989 (images), against the smallest class, disgust, with only 547 images—also, having a large amount of noise and non always apparent features. The dataset is classified as following: 0 = anger,1 = disgust,2 = fear,3 = happiness, 4 = sadness, 5 = surprise, 6 = neutral. From left to right (Blue is 0, Pink is 6)

![image](https://user-images.githubusercontent.com/65671410/170224099-f82eca5b-3d4f-4156-a851-87b6d45d4f1d.png)

![image](https://user-images.githubusercontent.com/65671410/170224371-ddf55470-8b41-4fc0-aa3f-38a0a10e9fc8.png)

![image](https://user-images.githubusercontent.com/65671410/170224417-b5747c92-5f69-42a1-b4bf-efd3d5403402.png)

![image](https://user-images.githubusercontent.com/65671410/170224439-60ed8a24-8d3f-4993-80fd-619b5bc68730.png)

![image](https://user-images.githubusercontent.com/65671410/170224453-23835db9-fbfa-468e-aec4-d3df5f2c19f0.png)

The latter image shows the case where the model weights, (hence, in simple words, what and how it learned) thus, the model does a wrong prediction. 

To build my model I used Keras, a high-level Neural Network library,
applied as a wrapper on Tensorflow. The latter, is an open-source library for machine learning having both high and low-level APIs. Tensorflow is used as a backend for the model. 

In the files you will find both .pynb version of the code and .py, as I used both Google Colab, which is an online environment that offers the use of CPU, GPU or TPU to train models, and Anaconda, using PyCharm as IDE installing CUDA and cudNN from Nvidia (for the use of GPU), to
test the efficiency on a local machine and the difference.


Validation accuracy 66% training on FER2013, tested using different datasets (e.g. CK+, JAFFE, random movies/series pictures, ...), or webcam input (photo, or real-time video).

This is a study based on the assumption of Universality of Emotions via Facial Expressions along cultures, backgrounds and ethnicities. One of the most protracted debates in biological and social science, as although many studies confirm this theory, on the other hand, many others do not believe in such a universality.
However, we will see that our model is somehow in part impacted by these non-universal expressions, being trained by a dataset such as FER2013.

The CNN structure is the following, you can change it depending on what you are training the model for:
![image](https://user-images.githubusercontent.com/65671410/170224259-f52ac7fd-8643-4e2b-9e2b-32da2c6e753b.png)

Py IDE Enviroment Code PyCharm enviroment created on
Anaconda, running Python 3.6 version, tensorflow-gpu 2.1.0, kerasgpu
2.3.1, CUDA 10.1.243, cudNN 7.6.5 versions.

If you don't want to use your machine, try using the Colab version (Jupyter), although not as complete as the local IDE version.
