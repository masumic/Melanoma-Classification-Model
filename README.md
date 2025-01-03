**Project Overview**

Web Application (Flask): 
The Flask web application offers an intuitive user interface where users can log in and upload images of skin lesions for classification.
Login: Users must log in with a predefined username and password (admin:admin) to access the image upload functionality.
Image Upload: After login, users can upload images of skin lesions, which are classified using the pre-trained model.
Results Display: The app will display the classification result (e.g., "Melanoma", "Basal Cell Carcinoma") and, for melanoma, request additional temperature data for further analysis.

Model (Keras/TensorFlow):
The classification model is built using TensorFlow and Keras. The model uses Convolutional Neural Networks (CNN) to classify the uploaded images into one of several lesion types, including melanoma.
The model is trained on a dataset of skin lesion images and is capable of detecting melanoma with high accuracy.
The trained model is saved as my_model.h5, and it is loaded for inference when processing uploaded images.


**File Structure**

app.py: The main Flask application that handles user login, file uploads, and displaying results.
model.py: The script used to define, train, and save the melanoma classification model.
templates/:
- login.html: The login page where users enter their credentials.
- index.html: The page for uploading images and displaying classification results.
