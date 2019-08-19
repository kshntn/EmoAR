# EmoAR
Repository for Facial Emotion Recognition Project for Udacity Secure and Private AI Challenge Scholarship 
Team:#sg_speak_german



# EmoAR – Facial expression recognition and Augmented Reality (AR)

A project by team #sg_speak_german:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Mateusz Zatylny, @Mateusz  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Berenice Terwey, @Berenice  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Akash Antony, @Akash Antony  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Calincan Mircea Ioan, @Calincan Mircea Ioan  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Joanna Gon, @Joanna Gon  

This project was planned and created as a team effort for the Facebook partnered Secure and Private AI challenge by Udacity and utilized the knowledge acquired in this course.

## Short description of our project: ##
EmoAR is a mobile AR application (mobile device with ARCore support is required) that aims to recognize human facial expression in real time and to overlay virtual content according to the recognized facial expression. For example: 
Depending on the predicted facial expression, EmoAR would display randomized famous quotes about the expression, in AR. (Quotes can motivate people to take positive changes in their life.)
The live AR camera stream of a mobile device (Android) is input to a segmentation tool (using tiny YOLO) that detects faces in the video frames in real time. 
The detected areas with a face are fed into a model that was trained on the public FER dataset (from a Kaggle competition 2013). 
The facial expression of the detected face is determined in real time by using our trained model. Depending on the model prediction (the output result), different virtual content overlays the face and adapts to the face's position. This virtual augmentation of a face is done with Augmented Reality (ARCore). 

Since ARcore is only supported by a small number of Android devices, we also deployed the model to a web app using Flask and Heroku, but without the AR feature. 

![project-diagram](https://user-images.githubusercontent.com/23194592/63302823-6d12fd00-c2de-11e9-9f0b-9a3cc274b243.jpg)

