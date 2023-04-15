"""
Description: This will create a parameters that we will use for 
             training the model - No. of epcohs, Optimizer, Learning rate, class lables
             saving the weights 
             Log file for logging

"""

class Config(object):
    IMAGE_HEIGHT=28
    IMAGE_WIDTH=28
    DATA_SIZE=4000
    CLASS_NAME=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    WEIGHT_FILENAME='FashionMNIST.h5'
    LOG_FILENAME='LOG_FASHION.txt'
    EPOCHS=125
    OPTIMIZER='RMSProp'
    LEARNING_RATE=0.001
    BATCH_SIZE=64

    def __init__(self):
        self.IMAGE_DIM = (self.IMAGE_HEIGHT, self.IMAGE_WIDTH)
