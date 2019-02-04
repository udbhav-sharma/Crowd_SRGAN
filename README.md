This package contains the entire codebase for Super Resolution.

Following is the structure of the code.

data- contains crowd images in two folders, train and test. Images are jpg format.

save- trained models are saved here.

data_utils.py- contains the definitions of data loader and image conversion functions. Main function is to return the pair of LR and HR Images

loss.py- The generator loss which is the perception loss is defined here

train.py- contains the training code

model.py- Defines the Generator and Discriminator. The generator is modified from the original paper and we do not upscale the input image to address the memory constraints.

unet- contains the definition of UNet architecture used as the generator. We do NOT use it for final experiments.

unet-train.py- similar to train.py but uses UNet generator.

test_image.py- contains code to generate super resolution images given input images.



To train the model, simply run:

python train.py

Hyperparameters:
  alpha- The downsampling and upsampling parameter for super resolution.

If you wish to change the network architecture to include upscaling of images, the Generator is defined using the upscaling factor and a flag tmode. Pass the upscaling factor as 2 or 4 or 8 and tmode = True when initializing the generator so that the model architecture is changed accordingly.

To test the trained model, configure the saved_model path in test_image.py and run:

python test_image.py
