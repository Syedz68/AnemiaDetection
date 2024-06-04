# Anemia Detection

This web application is designed to detect anemia from images using machine learning models. Built with Flask and TensorFlow/Keras, the application allows users to upload images, processes them, and provides a prediction on the presence of anemia.

## Features
- **Image Processing and Prediction**
  - *Image Upload*: Users can upload images through a web interface.
  - *Image Segmentation*: The application segments the uploaded image to focus on the region of interest.
  - *Mask Prediction*: Generates a mask for the segmented image.
  - *Conjunctival Region Extraction*: Extracts the conjunctival region from the image for further analysis.
  - *Anemia Prediction*: Predicts the presence of anemia based on the processed image.
- **Machine Learning Models**
  - *Custom Metrics and Loss Functions*: Implements dice_coef, dice_loss, and iou for model evaluation.
  - *Pre-trained Models*: Loads pre-trained models for mask prediction and anemia detection.
- **Asynchronous Processing**
  - *Async Image Processing*: Uses asynchronous functions to handle image processing tasks.

## Images of the web-app

![soft1](https://github.com/Syedz68/AnemiaDetection/assets/107263740/2419f5b3-db6f-43a6-bbab-ea61ae5668ec)

![soft2](https://github.com/Syedz68/AnemiaDetection/assets/107263740/eae0263f-caaa-4f78-b389-9efe8ec6e21e)
