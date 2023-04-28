# Imageclassification_Resnet_Moblinet-_NasNet
Resnet, mobilnet, and Nasnet are all included in this project, which have been optimized for Edge devices. There are two major ways to reach the goal: prunning and post train quantization.



Explain define funtion from project

1.generate_train_data_from_directory
This is a Python function that generates training data from images in a directory using the Keras ImageDataGenerator. It rescales the pixel values of images to be between 0 and 1, and generates batches of images and their labels from the directory.

The function takes in the following arguments:

train_data_dir: the path to the directory containing the training images.
image_target_size: the size to which the images will be resized.
batch_size: the number of images per batch.
channels: the number of color channels for the images (default is 3 for RGB).
class_mode: the type of label encoding (default is 'categorical' for one-hot encoding).
The function first creates an instance of the ImageDataGenerator with the specified rescaling factor. It then uses the flow_from_directory method of the generator to generate batches of training data from the specified directory, with the target size, batch size, and class mode specified.

The total number of images is calculated from the length of the generator, and the number of iterations needed to cover all the data is calculated from the batch size. The function then iterates over the generator and appends the images and their labels to separate lists, which are then converted to numpy arrays and returned.

This function can be useful for generating training data from a large number of images stored in a directory. The Keras ImageDataGenerator provides many useful preprocessing options, such as data augmentation and image normalization, which can improve the performance of machine learning models trained on image data.



2.inference_tflite
This is a Python function that performs inference on an integer quantized TensorFlow Lite model using a test dataset.

The function takes in the following arguments:

mode_path: the path to the integer quantized TensorFlow Lite model.
num_ingeter_test: the number of test images to use for inference.
The function first creates an instance of the TensorFlow Lite Interpreter with the specified model path, and allocates tensors for the input and output. It then prints the details of the input and output tensors.

The function then iterates over the test dataset, and for each batch, it performs inference on the input image using the Interpreter's set_tensor and invoke methods, and gets the output predictions using the get_tensor method. It also calculates the inference time for each image batch.

For each image, the function compares the true label with the predicted label, and keeps track of the number of correct predictions and the total number of images seen. It also prints the current accuracy after every 50 images processed.

Finally, the function prints the total number of images used, the accuracy of the model on the test dataset, and the average inference latency in milliseconds.

This function can be useful for evaluating the performance of an integer quantized TensorFlow Lite model on a test dataset. It measures both accuracy and inference latency, which are important metrics for evaluating the quality of a machine learning model.
