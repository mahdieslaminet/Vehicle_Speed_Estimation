# Vehicle_Speed_Estimation
Video-Based Vehicle Speed Estimation Using Speed Measurement Metrics

Vehicle Speed Estimation

Introduction:

This project aims to estimate vehicle speeds using video-based techniques and deep learning methods. By leveraging state-of-the-art models, we aim to achieve high accuracy in speed measurement, which is crucial for traffic monitoring and management.

Objectives:

•	Develop a robust vehicle speed estimation system using deep learning.

•	Minimize errors in speed measurement by addressing environmental and calibration factors.

•	Provide a comprehensive dataset and implementation details to ensure reproducibility and ease of use.

Dataset:

The dataset includes videos collected from various highway scenes, with annotations for vehicle speeds. Key considerations for the dataset:

•	Variety of lighting conditions

•	Different vehicle types and sizes

•	Multiple camera angles

Methodology:

Deep Learning Model
We employ a convolutional neural network (CNN) for feature extraction, followed by a regression model to estimate vehicle speeds. The model is trained on the annotated dataset to learn the correlation between visual features and vehicle speeds.

Error Correction:

To improve measurement accuracy, we implement robust calibration procedures and compensate for environmental factors such as varying lighting conditions and camera angles. These steps are crucial to minimize errors and enhance the reliability of the speed estimation system.

Installation:

To set up the project, follow these steps:

1.	Clone the repository
  https://github.com/mahdieslaminet/Vehicle_Speed_Estimation.git
2.	Install the required dependencies:
  pip install -r requirements.txt

Usage:

1.	Prepare the dataset by organizing the videos and annotations in the specified format.

2.	Train the model using the provided training scripts:

   python train_model.py --dataset_path path_to_dataset
  	
3.	Evaluate the model on test data:

python evaluate_model.py --dataset_path path_to_test_dataset

Results:

The trained model demonstrates high accuracy in speed estimation with minimal errors. Detailed results and performance metrics can be found in the Results section.

https://drive.google.com/file/d/1zYLwgu20Pg3m-PnWeyT0HplIPsfrs6rM/view?usp=drive_link

Examples:

We provide example scripts and notebooks to help you understand how to use the model and interpret the results. These examples cover:

•	Loading and preprocessing the dataset

•	Training the model

•	Evaluating the performance

•	Visualizing the results

Videos:

All instructional videos are publicly available on Google Drive. Links to these videos are provided below:

•	Training Process https://drive.google.com/file/d/1zYLwgu20Pg3m-PnWeyT0HplIPsfrs6rM/view?usp=drive_link

•	Evaluation Process https://drive.google.com/file/d/1zYLwgu20Pg3m-PnWeyT0HplIPsfrs6rM/view?usp=drive_link

•	Result Visualization https://drive.google.com/file/d/1zYLwgu20Pg3m-PnWeyT0HplIPsfrs6rM/view?usp=drive_link

Contributing:

We welcome contributions from the community. If you have suggestions, bug reports, or feature requests, please open an issue or submit a pull request. Make sure to follow the contributing guidelines.

This project implements a system for estimating vehicle speed in a video using computer vision and machine learning techniques.

Dependencies:

The project requires the following Python libraries:

OpenCV (cv2)
NumPy (np)
Deep SORT
YOLOv3 object detection model
scikit-learn (linear regression and MLPRegressor)
TensorFlow (Keras for CNN and RNN models)
Installation:

You can install these libraries using pip:


pip install opencv-python numpy deep_sort yolov3 tensorflow scikit-
learn

https://gemini.google.com/faq#coding

Project Overview:

The project performs the following steps:

Video Loading and Parameter Definition:

Loads the video file using OpenCV.
Defines parameters like frames per second (FPS), distance between reference lines, and line coordinates for speed measurement.
Object Detection and Tracking:

Uses YOLOv3 to detect vehicles in each frame.
Applies Deep SORT to track detected vehicles across frames and maintain unique IDs.
Wheel Tracking:

Employs the Good Feature to Track (GFTT) algorithm to track specific points on the vehicle (e.g., wheels) within a designated side area.
Speed Estimation:

Defines different machine learning models (Linear Regression, MLPRegressor, CNN, RNN) for speed estimation.
Trains these models on pre-collected data (replace example data with actual training data).
Tracks features (e.g., GFTT points) within the side area and feeds them to the chosen model for speed prediction.

Results Visualization:

Overlays reference lines on the video frame for speed measurement.
Displays the estimated speed for each tracked vehicle.
Visualizes YOLOv3 detection bounding boxes and Deep SORT track IDs.
Running the Project
Download the project files: Clone or download the project repository containing the Python script and any necessary model files.
Set up the video path: Modify the video_path variable in the script to point to your video file.
Train the machine learning models (optional): If you want to use a different model or train on your own data, modify the model definition and training sections in the script. 4. Run the script: Execute the Python script using a command like python main.py.

Output:

The script displays the video with speed information overlaid on each frame. It also calculates and prints the average speed of all tracked vehicles.

Additional Notes:

The script currently uses example data for the machine learning models. Replace this with actual training data for improved accuracy.
You can experiment with different machine learning models and parameters for speed estimation.
Ensure the YOLOv3 model is trained to detect vehicles and potentially wheels for better tracking.

Credits:

Deep SORT:

https://github.com/theAIGuysCode/yolov3_deepsort

YOLOv3:

https://github.com/eriklindernoren/PyTorch-YOLOv3

Link to the description of the code source:

https://drive.google.com/file/d/1zYLwgu20Pg3m-PnWeyT0HplIPsfrs6rM/view?usp=drive_link

Link to the description in the video:

https://drive.google.com/file/d/1_LgIY1EIBVe6reEPUcCueDBzQT3f0q5U/view?usp=drive_link

References:

•	Yunchao Zhang et al., "Deep Learning Based Vehicle Speed Estimation on Highways"

•	Kester Robert and Richard Bose, "Estimation of Optical Speed Measurement Error for Traffic"

•	Additional relevant papers and documentation.

License:

This project is licensed under the MIT License. See the LICENSE file for more details.

Contact:

For any inquiries or further information, please contact aminba81@gmai.com.

