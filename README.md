# handwritten-digits-MNIST

This is Selena's first attempt and creating a neurel network for the the handwritten-digits MNIST dataset.

OVERVIEW:

This is Selena's attempt at creating a neural network semi-from scratch
for the hand-written digits MNIST data set.

ABOUT THE DATASET: (QMNIST)

Note: I use the qmnit dataset which is the hand-written MNIST data set
with 60,000 images. It is part of the official torchvision datasets in 
pytorch documentation.

MAIN BUGS:

1. The labels for pyplot are messed up. Instead of indexing image 0 with
   label 0, every image has label 0 showing. I played around with this to
   try to figure it out so right now every image is labeled "happy" 
   instead of the correct #.

2. The accuracy is 100.0% every single time. Is this an error? 
   Note that the loss gets smaller each epoch. 

3. I haven't figured out how to properly push to github while using a 
   .gitignore file for the model_weights.pth file (which saves the model
   weights). I solved this by commenting out my code which saves & loads the
   model & deleting the model_weights.pth file. To save & load the model,
   simply uncomment the code & a model_weights.pth file will download. 
   question: How can I solve this? Should I use a .gitignore file whenever 
   uploading to github? I don't really want the file locally since it's just a 
   tutorial & since it takes up a lot of space.

NOTES:

I also have questions sprinkled throughout my code, noted with 'question'
