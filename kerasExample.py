from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.regularizers import l1, activity_l1
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import tensorflow as tf

# convert flatten data to 48*48 matrix
def reshapeTo48and48(dataset):

    #extract pixels value from original pandas dataframe
    pixels_values = dataset.pixels.str.split(" ").tolist()

    #initialize np array.
    npDataSet = np.empty((len(pixels_values), len(pixels_values[0])), dtype=float)

    row = len(pixels_values[0])
    # assign pixels_values to np array
    for x in range(len(pixels_values)):
        if len(pixels_values[x]) == row:
            npDataSet[x] = np.array(pixels_values[x], dtype=float)


    #convert pixels of each image to 48*48 formats
    images = []

    for image in npDataSet:
        images.append(image.reshape(48, 48))
    return np.array(images, dtype=float)

def sample_model():
    #initial model
    model = Sequential()
    # add dropout to reduce overfitting
    # model.add(Dropout(0.2, input_shape=(48, 48, 1)))
    #with 64 filters, 5*5 for convolutional kernel and activation 'relu'
    model.add(Convolution2D(64, 5, 5, input_shape=(48, 48, 1), activation='relu'))
    #pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, 5, 5, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))


    model.add(Flatten())
    #fully connected layer
    model.add(Dense(600, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(num_classes, W_regularizer=l1(0.01),
                    activity_regularizer=activity_l1(0.01),
                    activation='softmax'))
    return model

def kerasToTF(filePath):
    weight_file = filePath
    num_output = 1
    write_graph_def_ascii_flag = True
    prefix_output_node_names_of_final_network = 'output_node'
    output_graph_name = 'constant_graph_weights.pb'

    output_fld = 'tensorflow_model/'
    if not os.path.isdir(output_fld):
        os.mkdir(output_fld)
    weight_file_path = weight_file

    K.set_learning_phase(0)
    net_model = load_model(weight_file_path)

    pred = [None] * num_output
    pred_node_names = [None] * num_output
    for i in range(num_output):
        pred_node_names[i] = prefix_output_node_names_of_final_network + str(i)
        pred[i] = tf.identity(net_model.output[i], name=pred_node_names[i])
    print('output nodes names are: ', pred_node_names)

    sess = K.get_session()

    if write_graph_def_ascii_flag:
        f = 'only_the_graph_def.pb.ascii'
        tf.train.write_graph(sess.graph.as_graph_def(), output_fld, f, as_text=True)
        print('saved the graph definition in ascii format at: ', osp.join(output_fld, f))

    from tensorflow.python.framework import graph_util
    from tensorflow.python.framework import graph_io
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
    graph_io.write_graph(constant_graph, output_fld, output_graph_name, as_text=False)
    print('saved the constant graph (ready for inference) at: ', osp.join(output_fld, output_graph_name))

if __name__ == '__main__':
    # read data files.
    train = pd.read_csv("training.csv")
    test = pd.read_csv("testing.csv")
    # show training row data
    print(train.head())
    # show number of emotions based on individual labels.
    print(train.emotion.value_counts())

    # print number of training and testing data
    print("The number of traning set samples: {}".format(len(train)))
    print("The number of testing set samples: {}".format(len(test)))

    train_images = reshapeTo48and48(train)
    test_images = reshapeTo48and48(test)

    # reshape to [# of samples][width][height][pixels] for tensorflow-keras input format
    train_images = train_images.reshape(train_images.shape[0], 48, 48, 1).astype('float32')
    test_images = test_images.reshape(test_images.shape[0], 48, 48, 1).astype('float32')

    # normalize the data
    train_images = train_images / 255
    test_images = test_images / 255

    # One hot encode outputs: change target(emotion) values to input format with one-hot encode
    train_targets = np_utils.to_categorical(train.emotion.values)
    test_targets = np_utils.to_categorical(test.emotion.values)

    # set number of prediction classes (number of output classes, if binary classification, you should set num_classes = 2)
    num_classes = test_targets.shape[1]

    # define training model here.
    model = sample_model()

    # set information for saving model
    filename = "MyBestModel.hdf5"
    check_point = ModelCheckpoint(filename, monitor='val_acc', verbose=2, save_best_only=True,
                                  mode='max')
    callbacks_list = [check_point]

    # training model
    # compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # begin to train and save training history
    history = model.fit(train_images, train_targets, validation_data=(test_images, test_targets),
                        nb_epoch=2, batch_size=80, callbacks=callbacks_list, verbose=2)


    # plot history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Accuracy-Iteration Graph')
    plt.ylabel('Accuracy')
    plt.xlabel('Iteration(epoch)')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # evaluate the result on the testing data
    scores = model.evaluate(test_images, test_targets, verbose=0)
    print("%s: %.2f%%" % ("acc", scores[1] * 100))

    # load trained model from hdf5 file generated by keras.
    model.load_weights('myBestModel.hdf5')

    # this function compile your model to tensorflow format --> .pb file, use for classification in your Android project.
    # ref: https://github.com/amir-abdi/keras_to_tensorflow
    kerasToTF('MyBestModel.hdf5')

    # Since when you trying to load .pb file in android studio, you need to specify the name for both input nodes and output
    # nodes, the kerasToTF function should print the output nodes, however you need to get the input nodes as well.
    # One option is to use tensor graph, or you can run the code below should print the list of nodes from left to right
    # the first node on the left will be the name for your input node in this tensorflow model.
    gf = tf.GraphDef()
    gf.ParseFromString(open('tensorflow_model/constant_graph_weights.pb', 'rb').read())
    df = [n.name for n in gf.node]
    print(df)


