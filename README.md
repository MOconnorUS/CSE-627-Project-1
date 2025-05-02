# CSE 627 Project 1 - Learned Skeletonization
This project utiliizes a UNet architecture to take grey scaled images of roads and skeletonize them via double convolutions and weighted binary cross entropy loss. However, along with this baseline case there are also other methodologies for skeletonizing implemented. There is an Iterative Thinning wrapper model to utilize the UNet to iteratively thin the input data. Also, dice loss is implemented as well to switch from weighted binary cross entropy loss along with the option of changing the UNet activation function from ReLU to LeakyReLU.

## How to Run
To run this project you must first have python installed. This project was created using pip and thus the setup.bat file provided utilizes pip aswell. If you use conda you will need to change the manner in which the imports are installed. 
Once you clone the repository, it is recommended you establish a virtual environment to manage the imports solely for this project.
To install everything on windows simply cd into the cloned repository and enter the following into your terminal: setup.bat
This will install all the libraries used for this project.
Once everything is installed, you must cd into the models folder and then type the following into your terminal: python train.py
This will run the model on its current training parameters, architecture, and loss function.
