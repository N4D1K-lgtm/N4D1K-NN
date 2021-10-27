# Basic Neural Network for MNIST Dataset Written in Python

The entire purpose of this is to show my friends that wont care (and also write a paper)

## Numpy CNN
I handwrote everything but the adam optimizer but almost none of it is my code, followed lots of youtube videos explaining everything and giving examples.
If you have questions I can explain almost everything (or can try to), besides the stuff I specifically said I didn't understand in code comments.
If you're dave, dont judge my code. If you're not dave... think whatever you like. Feel free to contribute to add to a branch or message me directly.
(nate wrote stuff for importing MNIST images)

This can take several minutes to train and run (especially if you try experimenting with lots of layers/neurons/epochs)
This version hit a max of about 91% with an adam optimizer, no dropout, regularization and 300 iterations (or epochs). 
I never implented batches so it trains on all 60000 images every epoch. 

### Dependencies for Numpy Version:
`pip install python-MNIST`  
`pip install -U scikit-learn`   
`pip install numpy`  
Go to `/Numpy/bin` and run:  
`./get_mnist_data.sh`

## KERAS CNN
This is almost entirely from a book with some negligable QOL changes and different variable names (I also changed the optimizer). Writing the network from scratch helped me understand exactly what was going on. KERAS is a machine learning python library built on top of tensorflow/theano. Should check it out if you find this stuff interesting. (https://keras.io/)

Due to the magic of CUDA parallelization/actually well writting python libraries, compile time is significantly faster. (like 14 seconds on a GTX 1070/I5 4690k)
This CNN consistently hits 98% on the training set. (10000 images) It does this in just 20 epochs through randomly shuffled batches of 128 images. 
A 30% dropout rate increases the total accuracy on the training set by about 2%. 

GUI might happen if I don't get distracted with other cool stuff. (no promises)

### Dependencies for Keras Version:
`pip install keras`

