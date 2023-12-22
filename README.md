
# **LANE DETECTION USING ENET**
Lane detection is a crucial computer vision task that involves identifying the boundaries of driving lanes in an image or video of a road scene. It plays a vital role in various applications, particularly in the realm of autonomous vehicles and advanced driver-assistance systems (ADAS).  Convolutional neural networks (CNNs) are now the dominant approach for lane detection. Trained on large datasets of road images, these models can achieve high accuracy even in challenging situations.  In this we implemented ENET architecture which is a deep learning algorithm widely used for image segmentation.
## ENET(EFFICIENT NET)
ENet (Efficient Neural Network) is a deep neural network architecture designed for real-time semantic segmentation tasks.    ENet is significantly faster than other popular semantic segmentation architectures, such as VGG and ResNet. This is due to its use of several techniques to reduce the number of parameters and floating-point operations (FLOPs) required. For example, ENet uses a smaller number of convolutional filters and smaller kernel sizes than other architectures. Additionally, ENet uses a technique called "bottleneck residual connections" to reduce the number of FLOPs required for each layer.

Overall, ENet is a powerful and efficient architecture for real-time semantic segmentation tasks. It is a good choice for applications where low latency and memory footprint are important considerations.


## TuSimple Dataset
The TuSimple dataset is a large-scale dataset for autonomous driving research, focusing on lane detection and perception tasks. It's widely used in computer vision and autonomous driving communities for benchmarking and developing algorithms.

The TuSimple dataset consists of 6,408 road images on US highways. The resolution of image is 1280×720. The dataset is composed of 3,626 for training, 358 for validation, and 2,782 for testing called the TuSimple test set of which the images are under different weather conditions.



## ENET Architecuture 

![ENet Architecture](https://github.com/GirishChowdary0208/Lane_Detection_Using_ENet/assets/92716279/d926b192-ee0d-406a-8efa-bd4a4e02498c)

## Downloads :    
Download the Dataset Here: [TuSimple](https://www.kaggle.com/datasets/manideep1108/tusimple)



Checkout the Kaggle Link for this project : [Kaggle](https://www.kaggle.com/code/rangalamahesh/lane-detection-using-enet)
## Getting Started 

To run this project you can download the ENET.ipynb file provided in the repository and the dataset from the download section and can implement the whole process by following the instructions in the [Kaggle Link](https://www.kaggle.com/code/rangalamahesh/lane-detection-using-enet).  Below are the basic Requirements to run the code 
  - ![](https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
  - GPU
  - CUDA

I choose Kaggle to implement this because it provides inbuilt GPU accelerator which accelerate the training process, I used  GPU P100 to implement this.  You can also choose google colab to run this, google colab also provides inbuilt GPU accelerator which fast up the training process much faster that using CPU.
## Training the Model

To train this model I used  GPU P100 accelerator which accelerated my trained process much more faster than using CPU.  In my model training process the training Epochs are 32, batch size is 8 with Adam Optimizer with learning rate of 0.00005 and the process went well with higher accuracy and low loss. 

I used  [TuSimple](https://www.kaggle.com/datasets/manideep1108/tusimple) to implement this project and the training process went very well with an overall good accuracy and low loss. 



### Test 

You can download the weights file ENET.pth file and directly test it for predictions.  

Also find the inference_ent.ipynb file which contained the testing or inference code.

To test the code
```bash
  Download the inference_enet.ipynb file and load the model weights
  ENET.pth path  and provide the testing image path in the inference code. 
  By running the inference_enet.ipynb file you can visualize the plot of the predictions.
```

## METRICS VISUALIZATION

![ENet Metrics](https://github.com/GirishChowdary0208/Lane_Detection_Using_ENet/assets/92716279/d52b7c1a-d4f7-4cbd-96be-f82d95cda9a9)


The Above graph visualize the metrics during the training process, it shows the graph showing Training & Validation Loss and Training & Validation Accuracy with the staring value and ending value.  The graphs shows the gradual decrease in the loss function and gradual increase accuracy as shown in the visualization.

You can also check the TensorBoard logs to visualize the metrics and the layers in the Architecture.  Here are the TensorBoard visualization of graphs:

![ENet Metrics G-1](https://github.com/GirishChowdary0208/Lane_Detection_Using_ENet/assets/92716279/bc51c5df-41dd-4ab4-a3db-d2282184391a)

The Above graph visualize the Binary Accuracy of ENet model visualized using TensorBoard.

![ENet Metrics G-2](https://github.com/GirishChowdary0208/Lane_Detection_Using_ENet/assets/92716279/0abd296e-d998-44a0-9104-9f4932dde372)

The Above graph visualize the Binary Loss of ENet model visualized using TensorBoard.

![ENet Metrics G-3](https://github.com/GirishChowdary0208/Lane_Detection_Using_ENet/assets/92716279/1d3296a4-b844-40ab-95b6-61b9a7d29fd3)

The Above graph visualize the Instance Loss of ENet model visualized using TensorBoard.

To run the TensorBoard logs follow the command in your Terminal:
```bash
tensorboard --logdir=path/to/your/logs/directory
```
After running the command, open your web browser and go to http://localhost:6006 to access the TensorBoard interface. You'll be able to navigate through the different tabs to explore the data recorded in the tensorboard v2 file.
## Predictions 

![ENet output1](https://github.com/GirishChowdary0208/Lane_Detection_Using_ENet/assets/92716279/d6d6e520-10c9-48e8-8297-6a510ac3b364)

![ENet output2](https://github.com/GirishChowdary0208/Lane_Detection_Using_ENet/assets/92716279/56b7878b-a29b-44c0-9450-95cc66fda1c3)

![ENet output3](https://github.com/GirishChowdary0208/Lane_Detection_Using_ENet/assets/92716279/8a37e31a-a8f1-404e-94b6-bd7f5bd2ac94)

![ENet output4](https://github.com/GirishChowdary0208/Lane_Detection_Using_ENet/assets/92716279/48084fa7-7241-42c6-82c6-2a8811aa83b0)

I tested the Predictions on the inference code by loading the saved .pth weights file and testing it on the new images.  The model predictions came out to be good as shown in the figures.

## 🔗 Connect with me
[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/GirishChowdary0208)

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/girish-chowdary-919b6522b/)

[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/_GirishChowdary)

[![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/girishchowdary22)

[![ResearchGate](https://img.shields.io/badge/ResearchGate-00CCBB?style=for-the-badge&logo=ResearchGate&logoColor=white)](https://www.researchgate.net/profile/Girish-Chowdary)

## 🚀 About Me

I am Girish Chowdary, an enthusiastic and versatile individual deeply passionate about the realm of technology and its endless possibilities.

- 🔭 Embarking on the journey of knowledge at Centurion University of Technology and Management for a BTech in Computer Science and Engineering.

- 🌱 Exploring the vast landscapes of Artificial Intelligence, Machine Learning, Deep Learning, Neural Networks, and various Programming Languages to unlock boundless potential.

- 👨‍💻 Showcasing my passion through innovative projects, all neatly organized at https://github.com/GirishChowdary0208.

- 📫 Connect with me via email at mupparjugirishchowdary@gmail.com to share ideas, collaborate, or discuss the limitless possibilities of technology.

