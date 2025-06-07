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


## Run the Project:

I trainined this network using Google Co-lab with GPU: NVIDIA A100-SXM4-40GB. You can run the BowlDetectorCode.ipynb jupyter notebook.


## Model:

The Model I used is Faster R-CNN with ResNet-50 backbone + 5-level FPN.

I selected this model based on this research: <br />
![ModelResearch](https://github.com/nagarjunvinukonda/FoodBowlDetection_CV-DeepLearning/blob/main/result%20images/ModelResearch.png) <br />


### Training performance: <br />

With 4-Fold Cross Validation and its inferences<br />
![TrainingPerformace2](https://github.com/nagarjunvinukonda/FoodBowlDetection_CV-DeepLearning/blob/main/result%20images/TrainingPerformace2.png) <br />
![TrainingPerformace](https://github.com/nagarjunvinukonda/FoodBowlDetection_CV-DeepLearning/blob/main/result%20images/TrainingPerformace.png) <br />

## My result:
![1](https://github.com/nagarjunvinukonda/FoodBowlDetection_CV-DeepLearning/blob/main/result%20images/1.png) <br />
![2](https://github.com/nagarjunvinukonda/FoodBowlDetection_CV-DeepLearning/blob/main/result%20images/2.png) <br />
![3](https://github.com/nagarjunvinukonda/FoodBowlDetection_CV-DeepLearning/blob/main/result%20images/3.png) <br />
![4](https://github.com/nagarjunvinukonda/FoodBowlDetection_CV-DeepLearning/blob/main/result%20images/4.png) <br />
![5](https://github.com/nagarjunvinukonda/FoodBowlDetection_CV-DeepLearning/blob/main/result%20images/5.png) <br />

Precision Vs Recall: <br />
![Precision Vs Recall](https://github.com/nagarjunvinukonda/FoodBowlDetection_CV-DeepLearning/blob/main/result%20images/PVR.png) <br />


## Summary of this project: 

1. I re‐labeled all training/test images in YOLO format to ensure consistent empty vs. full classes and corrected all bounding‐box placements (including rotated bowls). 
2. I implemented a Faster R-CNN (ResNet-50 + FPN) detector in PyTorch, fine‐tuned from COCO weights, with data augmentations (flip + color jitter) and a GIoU loss term to improve localization.
3. I performed 4-fold cross‐validation (stratified to avoid near‐duplicate leakage), each fold trained for 30 epochs.
4. I reported per‐fold mAP@0.5 / mAP@[0.5:0.95] metrics and the final held‐out test scores (Test mAP@0.5 ≈ 0.38).
5. I packaged our deliverables:
o final.ipynb containing all code cells (data loading, training, validation, evaluation, ONNX export). <br />
o fasterrcnn_bowl_best.pth (PyTorch weights) <br />
o bowl_detector.onnx (ONNX model for inference) <br />
o Precision–Recall curve and GT vs. prediction images saved as PNG. <br />
o This detailed report (no page limit) and the earlier one‐page summary. <br />
6. I outlined extensions for predicting bowl orientation and recognizing ingredients, including multi‐task architectures and semi‐supervised bootstrapping. <br />

By following these instructions, anyone can reproduce our results in Colab, inspect the example detections, and use our ONNX model on‐device (e.g., a Jetson Orin or a CPU laptop). The next steps to improve generalization would be to collect 1 k+ diverse images, annotate via a human‐in‐the‐loop pipeline, and explore one‐stage detectors (YOLOv5/7) or segment‐based approaches for pixel‐level precision.
