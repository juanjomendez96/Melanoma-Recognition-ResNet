# Melanoma Recognition 
This is a final bachelor's project for the Informatics Engeniering at Córdoba's University, Spain. Here we can observe all the related files for create a residual neural network for skin lesion classification. This project has been created trying to simulate the work of the article [here](Articles/Article_Melanoma_Recognition.pdf). 
In order to use this program, you have to:
1. Download images of skin lesions and the metadata from [ISIC gallery](https://www.isic-archive.com/#!/topWithHeader/wideContentTop/main)
2. (Optional) Preprocess images and then start the process of training the model
3. Train the model. To do this, you have to run the following command:
> python3 main.py -d [path-hdf5-file] -tt [name-hdf5-file] -lr [learning-rate] -e [epochs] -p [patiente] -o [optimizer-type] -b [batch-size]
###### For more help, please, run python3 main.py -h
4. Once the training process has been completed, you can check the model with the [jupyter notebook](src/Model_check.ipynb) that I have created. In here you will be able to load the weights of your best model and then, check if the network works as desired.
5. At the end, you will have two images, the confusion matrix and the train test loss figure, with a folder that keeps the events of the model and the weights of it. You can check the events using tensorboard as follows:
> tensorboard --logdir='logs/'
