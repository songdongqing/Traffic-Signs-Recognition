from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.merge import concatenate
import cv2
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import utils
from keras.models import load_model
from util import *


np.random.seed(23)

# 对图片进行预处理
def preprocess_features(X, mean_img, std_img):   # (39209, 32, 32, 3)
    # convert from RGB to YUV
    X = np.array([np.expand_dims(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YUV)[:, :, 0], 2) for rgb_img in X])

    # 直方图均衡化
    X = np.array([np.expand_dims(cv2.equalizeHist(np.uint8(img)),2) for img in X])

    X = np.float32(X)

    # 图像标准化
    X -= mean_img
    X /= std_img
    return X

# 图像标准化
def get_mean_std_img(X):
    # convert from RGB to YUV
    X = np.array([np.expand_dims(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YUV)[:, :, 0], 2) for rgb_img in X])
    X = np.array([np.expand_dims(cv2.equalizeHist(np.uint8(img)), 2) for img in X])
    X = np.float32(X)

    mean_img = np.mean(X,axis=0)
    std_img = (np.std(X,axis=0)+ np.finfo('float32').eps)
    return mean_img,std_img

# 对于任意一张图片，进行数据增强之后显示的图片
def show_samples_from_generator(image_datagen, X_train, y_train):
    # take a random image from the training set
    img_rgb = X_train[0]

    # plot the original image
    plt.figure(figsize=(1, 1))
    plt.imshow(img_rgb)
    plt.title('Example of RGB image (class = {})'.format(y_train[0]))
    plt.show()

    # plot some randomly augmented images
    rows, cols = 4, 10
    fig, ax_array = plt.subplots(rows, cols)
    for ax in ax_array.ravel():
        augmented_img, _ = image_datagen.flow(np.expand_dims(img_rgb, 0), y_train[0:1]).next()
        ax.imshow(np.uint8(np.squeeze(augmented_img)))
    plt.setp([a.get_xticklabels() for a in ax_array.ravel()], visible=False)
    plt.setp([a.get_yticklabels() for a in ax_array.ravel()], visible=False)
    plt.suptitle('Random examples of data augmentation (starting from the previous image)')
    plt.show()


def get_image_generator():
    # create the generator to perform online data augmentation
    image_datagen = ImageDataGenerator(rotation_range=15.,
                                       zoom_range=0.2,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1
                                       )
    return image_datagen


def show_image_generator_effect():
    X_train, y_train = load_traffic_sign_data('./traffic-signs-data/train.p')

    # Number of examples
    n_train = X_train.shape[0]

    # What's the shape of an traffic sign image?
    image_shape = X_train[0].shape

    # How many classes?
    n_classes = np.unique(y_train).shape[0]

    print("Number of training examples =", n_train)
    print("Image data shape  =", image_shape)
    print("Number of classes =", n_classes)

    image_generator = get_image_generator()
    show_samples_from_generator(image_generator,X_train,y_train)

def get_model(dropout_rate = 0.0):
    input_shape = (32, 32, 1)

    input = Input(shape=input_shape)
    cv2d_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(input)
    pool_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(cv2d_1)
    dropout_1 = Dropout(dropout_rate)(pool_1)
    flatten_1 = Flatten()(dropout_1)

    dense_1 = Dense(64, activation='relu')(flatten_1)
    output = Dense(43, activation='softmax')(dense_1)
    model = Model(inputs=input, outputs=output)
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize model
    model.summary()
    return model


def train(model, image_datagen, x_train, y_train, x_validation, y_validation):
    # checkpoint
    filepath = "weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    callbacks_list = [checkpoint]
    image_datagen.fit(x_train)
    history = model.fit_generator(image_datagen.flow(x_train, y_train, batch_size=128),
                        # steps_per_epoch=5000,
                        steps_per_epoch=len(x_train)/128,
                        validation_data=(x_validation, y_validation),
                        epochs=20,
                        callbacks=callbacks_list,
                        verbose=1)

    # list all data in history
    print(history.history.keys())  # dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    with open('/trainHistoryDict.p', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    return history

def get_paper_model(dropout_rate=0.0):
    input_shape = (32, 32, 1)

    input = Input(shape=input_shape)
    cv2d_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(input)
    pool_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(cv2d_1)
    dropout_1 = Dropout(dropout_rate)(pool_1)
    flatten_1 = Flatten()(dropout_1)

    cv2d_2 = Conv2D(64,(3,3),padding='same',activation='relu')(dropout_1)
    pool_2 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(cv2d_2)
    cv2d_3 = Conv2D(64, (3, 3), padding='same', activation='relu')(pool_2)
    dropout_2 = Dropout(dropout_rate)(cv2d_3)
    flatten_2 = Flatten()(dropout_2)

    concat_1 = concatenate([flatten_1,flatten_2])

    dense_1 = Dense(64, activation='relu')(concat_1)
    output = Dense(43, activation='softmax')(dense_1)
    model = Model(inputs=input, outputs=output)
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize model
    model.summary()
    return model


def evaluate(model, X_test, y_test):
    score = model.evaluate(X_test, y_test, verbose=1)
    accuracy = score[1]
    return accuracy


def train_model():
    X_train, y_train = load_traffic_sign_data('./traffic-signs-data/train.p')

    # Number of examples
    n_train = X_train.shape[0]

    # What's the shape of an traffic sign image?
    image_shape = X_train[0].shape

    # How many classes?
    n_classes = np.unique(y_train).shape[0]

    print("Number of training examples =", n_train)
    print("Image data shape  =", image_shape)
    print("Number of classes =", n_classes)

    mean_img,std_img = get_mean_std_img(X_train)
    X_train_norm = preprocess_features(X_train,mean_img,std_img)  # (39209, 32, 32, 1)
    y_train = utils.to_categorical(y_train, n_classes)

    # split into train and validation
    VAL_RATIO = 0.2
    X_train_norm, X_val_norm, y_train, y_val = train_test_split(X_train_norm, y_train,
                                                                test_size=VAL_RATIO,
                                                                random_state=0)

    # model = get_model(0.0)

    # 使用论文里面的模型
    model = get_paper_model(0.0)
    image_generator = get_image_generator()
    train(model, image_generator, X_train_norm, y_train, X_val_norm, y_val)


if __name__ == "__main__":
    train_model()
    # show_image_generator_effect() # 查看数据增强产生的图片
    # 使用论文里面的model，loss: 0.1129 - accuracy: 0.9634 - val_loss: 0.0565 - val_accuracy: 0.9852


