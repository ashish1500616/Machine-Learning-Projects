import keras.models
from keras.models import model_from_json
import tensorflow as tf

def init_model_2(): 
    json_file = open('Model_Saved/model2/model2cnn_json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    #load woeights into new model
    loaded_model.load_weights("Model_Saved/model2/model2cnn_json.h5")
    print("Loaded Model from disk")

    #compile and evaluate loaded model
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #loss,accuracy = model.evaluate(X_test,y_test)
    #print('loss:', loss)
    #print('accuracy:', accuracy)
    graph=tf.get_default_graph()

    return loaded_model,graph