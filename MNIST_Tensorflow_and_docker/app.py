from flask import Flask, jsonify, request
import numpy as np
from fashion_mnist import classFashionMNIST, fashionMNIST
from fashion_config import Config as cfg
import logging
import absl.logging

app = Flask(__name__)

@app.route("/predict", methods = ['GET'])
def predict():
    pred = ""
    posted_data = request.get_json()
    test_image_num = posted_data['test_image_num']
    logging.info("In Predict")
    model_filename = cfg.WEIGHT_FILENAME
    pred = fashionMNIST.predict_data(test_image_num, model_filename)
    return jsonify(pred)

@app.route("/real", methods=['GET'])
def real():
    data = ""
    posted_data = request.get_json()
    test_image_num = posted_data['test_image_num']
    data = fashionMNIST.actual_data(test_image_num)
    return jsonify(data)

@app.route("/train", methods = ["POST", "GET"])
def train():
    history = ""
    posted_data = request.get_json()
    epochs = posted_data['epochs']
    if epochs == "":
        epochs = cfg.EPOCHS
    logging.info('Training')
    # Normalizing the data
    fashionMNIST.normalize_data()
    # Train the model
    history, model = fashionMNIST.train_model(cfg.WEIGHT_FILENAME,
                                                   epochs, cfg.OPTIMIZER,
                                                   cfg.LEARNING_RATE,
                                                   cfg.BATCH_SIZE)
    
    val_acc=str(np.average(history.history['val_acc']))
    acc=str(np.average(history.history['acc']))
    result={'val accuracy':val_acc, 'acc':acc}
    return jsonify(result)

if __name__ == '__main__':
    print('In logging')
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
    logging.basicConfig(filename=cfg.LOG_FILENAME, filemode='a', format='%(filename)s-%(asctime)s %(msecs)d- %(process)d-%(levelname)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S %p',
                        level=logging.DEBUG)
    # fashionMNIST = classFashionMNIST(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, cfg.DATA_SIZE, cfg.CLASS_NAME)
    fashionMNIST.normalize_data()
    history, model = fashionMNIST.train_model(cfg.WEIGHT_FILENAME, 
                                                   cfg.EPOCHS,
                                                   cfg.OPTIMIZER,
                                                   cfg.LEARNING_RATE,
                                                   cfg.BATCH_SIZE)
    
    app.run(debug=True)