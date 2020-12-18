# AMLS_assignment20_21
Student number: 19162034

Three functions in main repository:
1. data_prep: Preprocessing of the data, obtaining pictures and labels. 
For tasks A1 and A2 also extract features using shape_predictor_68_face_landmarks.dat

2. hyperparameter_set: Function to plot the accuracy against a value of a certain hyperparameter. 
For each one to be estimated, changes must be made to the value of the parameter and the algorithm.

3. main: Function to be executed. Only calls data_prep. Prints accuracy values for every task.

For every folder, one file with extended reasoning for code in main.py. 
Each file contains function to plot learning curve.
In folder A2, extra file used for setting how many features will be deleted. (A2_extra.py)

If desired to run code, insert dataset in folder Datasets (empty) with folder and file names as provided originally and run main.py
shape_predictor_68_face_landmarks.dat must be downloaded and imported into main repository. It is an imported function called by data_prep. 
Obtains 68 features for every image. Download link: https://drive.google.com/file/d/10yAaJnLxTvFsu6fboXy8tr7g_Dk-Z58a/view?usp=sharing

Packages needed: os, numpy, keras, sklearn, cv2, dlib
