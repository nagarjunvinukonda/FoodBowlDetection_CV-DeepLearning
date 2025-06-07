# FoodBowlDetection_CV-DeepLearning

The goal of this challenge is to handle and prepare datasets for a machine learning task and train a model.

The dataset consisting of 25 training images and 5 testing images. The task is to train (or fine-tune) a Machine Learning model that runs on device and will be used to detect bowls (the trays where food is deposited).

## Instructions
1. Data Analysis: <br />
a. Inspect the dataset for quality and consistency. Document any issues you find and how you would address them. <br />
b. How would you collect and annotate a larger dataset of images? <br />

2. Model Training: <br />
a. Choose a suitable deep learning model architecture for object detection. Justify your choice.<br />
b. Train the model using the images. Feel free to alter the dataset if needed.<br />

3. Evaluation: <br />
a. Evaluate the model performance using appropriate metrics for object detection. Discuss the results. <br />
b. Identify any potential limitations of the data and your model and suggest improvements. <br />

4. Documentation: <br />
a. Provide a brief report documenting your methodology, code, results, and any insights or challenges you encountered during the task.<br />

5. Extension: <br />
a. Discuss different alternatives to predict the bowl orientation. <br />
b. Discuss different alternatives to also output the ingredients in the bowl. <br />

## Deliverables
1. Source code for preprocessing, training, and evaluation. <br />
2. Trained model artifacts. <br />
3. A concise and detailed report (half a page to one page) covering methodology, results, and discussions. <br />


## Model:

The Model I used is Faster R-CNN with ResNet-50 backbone + 5-level FPN.


![image](https://github.com/user-attachments/assets/aba8e44b-421e-4e5e-971c-b26a5d6b10f5)


After training I got the following results: 


## My result:
![1](https://github.com/nagarjunvinukonda/FoodBowlDetection_CV-DeepLearning/blob/main/result%20images/1.png) <br />
![2](https://github.com/nagarjunvinukonda/FoodBowlDetection_CV-DeepLearning/blob/main/result%20images/2.png) <br />
![3](https://github.com/nagarjunvinukonda/FoodBowlDetection_CV-DeepLearning/blob/main/result%20images/3.png) <br />
![4](https://github.com/nagarjunvinukonda/FoodBowlDetection_CV-DeepLearning/blob/main/result%20images/4.png) <br />
![5](https://github.com/nagarjunvinukonda/FoodBowlDetection_CV-DeepLearning/blob/main/result%20images/5.png) <br />

Precision Vs Recall: <br />
![Precision Vs Recall](https://github.com/nagarjunvinukonda/FoodBowlDetection_CV-DeepLearning/blob/main/result%20images/PVR.png) <br />


