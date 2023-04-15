import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import logging
from tensorflow import keras

np.random.seed(1)

class classFashionMNIST:
    def __init__(self, height, width, data_size, class_name):
        '''
        Functionality: initializes the class
        Parameters:  sets the height, width of the image,  data size and class labels
        '''
        try:  
            self.height = height
            self.width = width
            self.data_size = data_size
            self.class_names = list(class_name)
            (self.train_images, self.train_labels) , (self.test_images, self.test_labels) = tf.keras.datasets.fashion_mnist.load_data()
            self.test_data = self.test_images
        except:
            logging.error("Error in init %s", sys.exc_info())

    def normalize_data(self):
        '''
        Functionality: Normalizes the images pixel intensity values by
                   scaling pixel values to the range 0-1 to centering and 
                   even standardizing the values.
        Parameters:  None
        '''
        try:
            logging.info("Normalizing data")
            # load train and test images and labels based on data size
            self.train_labels = self.train_labels[:self.data_size]
            self.test_labels = self.test_labels[:self.data_size]

            # #Normalize the data
            self.train_images = self.train_images[:self.data_size].astype('float32')/255
            self.test_images = self.test_images[:self.data_size].astype('float32')/255
            
            
            logging.info("Reshaping data")
            self.train_images = self.train_images.reshape((self.train_images.shape[0], self.width, self.height, 1))
            self.test_images = self.test_images.reshape((self.test_images.shape[0], self.width, self.height, 1))
        except:
            logging.error("Error", sys.exc_info())

    
    def create_model(self, optimizer, learning_rate):
        '''
        Functionality: Creates the deep learning model for multiclass classification
        Parameters:  optimizer - optimizers can be Adam, SGD or RMSProp
                    Learning_rate- learning rate of the optimizer
        '''   
        try:
            logging.info("Creating model")
            model = keras.Sequential() 
            model.add(keras.layers.Conv2D(filters=64, kernel_size = 2, padding = 'same', activation = 'relu', input_shape = (28,28,1)))
            model.add(keras.layers.MaxPooling2D(pool_size = 2))
            model.add(keras.layers.Dropout(0.3))
            model.add(keras.layers.Conv2D(filters=32, kernel_size = 2, padding = 'same', activation = 'relu'))
            model.add(keras.layers.MaxPooling2D(pool_size = 2))
            model.add(keras.layers.Dropout(0.3))
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(256, activation = 'relu'))
            model.add(keras.layers.Dropout(0.5))
            model.add(keras.layers.Dense(10, activation = 'softmax'))
            logging.info("Model Created")

            # creating optimizer based on the config
            opt = self.get_optimizer(optimizer, learning_rate)
            # Compiling the model
            model.compile(loss='sparse_categorical_crossentropy',
                          optimizer=opt,
                          metrics=['accuracy'])
            logging.info(" Model Compiled")
        
        except:
            logging.error(" Error during Model Creation - %s", sys.exc_info())
        
        finally:
            return model

    def get_optimizer(self,optimizer_name = 'RMSProp', learning_rate=0.001):
        '''
        Functionality: Creates the optimizers based on passed parameter and learning rate
        Parameters:  Optimizer_name - optimizers can be Adam, SGD or RMSProp
                    Learning_rate- learning rate of the optimizer
        '''  
        try:
            if optimizer_name=='Adam':
                #  If set to True, the AMSGrad method is used to calculate the adaptive learning rate.
                #  can improve network convergence and prevent premature learning rate decline
                optimizer = keras.optimizers.Adam(learning_rate = learning_rate, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)
            elif optimizer_name == 'SGD':
                optimizer = keras.optimizers.SGD(learning_rate = learning_rate, momentum=0.9)
            elif optimizer_name == 'RMSProp':
                optimizer = keras.optimizers.RMSprop()
            logging.info("Optimizer created %s", optimizer_name)
            return optimizer
        
        except:
            logging.error(" Error during visualization - %s", sys.exc_info())
    
    def train_model(self, filename, epochs, optimizer, learning_rate, batch_size):
        '''
        Functionality: Trains the deep learning multiclass classification model
        Parameters:  filename : File o save the trained weights
                     epochs : No. of epcohs to train the model
                     optimizer - optimizers can be Adam, SGD or RMSProp
                     Learning_rate- learning rate of the optimizer
                     Batch_size - batch_size of the dataset to train the model
        ''' 
        try:
            model = self.create_model(optimizer, learning_rate)
            logging.info("Model created")

            logging.info("Normalizing the data")
            self.normalize_data()
            logging.info(self.train_images.shape)

            logging.info("Training started")

            stop = keras.callbacks.EarlyStopping(monitor = 'val_accuracy' , patience=20, min_delta = 0.01, restore_best_weights=True)

            history = model.fit(self.train_images,
                                self.train_labels,
                                batch_size=batch_size,
                                epochs=epochs,
                                validation_data=(self.test_images, self.test_labels),
                                callbacks = [stop]
                              )
            logging.info(" Training finished")

            acc = history.history['accuracy'][-1]
            print(acc)
            val_acc = history.history['val_accuracy'][-1]
            print(acc)
            logging.info(" Model accurcay on train images : {:5.2f}".format(acc))
            logging.info("Accurcay too low for val {:5.2f}".format(val_acc))
            model.save(filename)
            logging.info("Model saved %s", filename)

            if acc < .8 or val_acc < 0.7:
                logging.warn("Accurcay too low {:5.2f}".format(acc) )
                logging.warn("Accurcay too low for val {:5.2f}".format(val_acc))

            return history, model

        except:
            logging.error(" Error during Model Creation - %s", sys.exc_info())

    def predict_data(self, test_image_num, filename):
        '''
        Functionality: predicts the data for  multiclass classification model
        Parameters: test_image_num - index of the test image that we want to predcit 
                filename : File containing  the trained weights
                    
        ''' 
        try:
            logging.info("Predciting the data for %d", test_image_num)
            test_img = self.test_images[test_image_num].reshape((1, self.width, self.height, 1))
            test_img = test_img.astype('float')/255
            model = keras.models.load_model(filename)
            logging.info("Loaded the trained weights from %s", filename)
            pred = model.predict(test_img)
            pred = np.argmax(pred)
            logging.info("Predicted class  %s",self.class_names[pred])
            return self.class_names[pred]
        
        except:
            logging.error(" Error during Model predcition - %s", sys.exc_info())

    def actual_data(self, test_image_num):
        '''
        Functionality: Retrives the actual class for the test image based on the index passed
        Parameters: test_image_num - index of the test image that we want to predcit 
        '''
        return self.class_names[self.test_labels[test_image_num]]
    
    def visualize(self, test_image_num):
        '''
        Functionality: Retrives the actual class for the test image based on the index passed
        Parameters: test_image_num - index of the test image that we want to predcit 
        '''
        try:
            plt.imshow(self.test_images[test_image_num])
            plt.title(str(self.actual_data(test_image_num)))
            plt.show()

        except:
            logging.error(" Error during visualization - %s", sys.exc_info())

