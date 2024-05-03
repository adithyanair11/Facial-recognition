# Facial-recognition
In this project we implement a facial identification and emotion recognition model, in real-time via the webcam attached to a computer system. This is done using Tensorflow and Keras to implement the Convolutional Neural Network, as well as OpenCV to access the frames from a webcam and perform various other Computer Vision tasks.

The Facial Emotion Recognition model is created from the notebook - Facial_Expression_Recognition_Model.ipynb. It uses Deep Learning tools and Convolutional Neural Network to classify emotions into 6 categories - Anger, Neutral, Fear, Happy, Sad and Surprise. The model is then stored as model.h5

The Facial Identification is performed via the create_data.py file where a webcam captures 30 frames of a subjects face and stores it in a desired location on the system.

main_new.py then integrates both the features as a single program and performs both operations in real-time by accessing the webcam and testing the functions on a subject.
