from data.DataProcessing import *

import keras.backend as K
from keras.models import Model, Sequential
import keras
from keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, Dense, Input, Reshape, Lambda, Dropout, Concatenate, Add, Activation
from keras.optimizers import Adam

IMAGES_PATH = ''
LOGO_PATH = ''

imgs_arr = load_images(IMAGES_PATH)
logo_names = load_logos(LOGO_PATH)

train_gen = data_gen(imgs_arr[len(imgs_arr)//5:], logo_path, logo_names[len(logo_names)//5:], 16)
val_gen = data_gen(imgs_arr[:len(imgs_arr)//5], logo_path, logo_names[:len(logo_names)//5], 4)

def mask_model_func():

  mask_model_in = Input(shape=(256, 256, 3))

  mask_model = (Conv2D(64, 3, padding='same', activation='relu'))(mask_model_in)
  mask_model = (Conv2D(64, 3, padding='same', activation='relu'))(mask_model)
  mask_model = (Conv2D(64, 3, padding='same', strides=2))(mask_model)

  mask_model = (Conv2D(128, 3, padding='same', activation='relu'))(mask_model)
  mask_model = (Conv2D(128, 3, padding='same', activation='relu'))(mask_model)
  mask_model = (Conv2D(128, 3, padding='same', strides=2))(mask_model)

  mask_model = (Conv2D(256, 3, padding='same', activation='relu'))(mask_model)
  mask_model = (Conv2D(256, 3, padding='same', activation='relu'))(mask_model)
  mask_model = (Conv2D(256, 3, padding='same', strides=2))(mask_model)

  mask_model = (Conv2D(512, 3, padding='same'))(mask_model)


  mask_model = (Conv2DTranspose(256, 3, padding='same', strides=2))(mask_model)
  mask_model = (Conv2D(256, 3, padding='same', activation='relu'))(mask_model)
  mask_model = (Conv2D(256, 3, padding='same', activation='relu'))(mask_model)

  mask_model = (Conv2DTranspose(128, 3, padding='same', strides=2))(mask_model)
  mask_model = (Conv2D(128, 3, padding='same', activation='relu'))(mask_model)
  mask_model = (Conv2D(128, 3, padding='same', activation='relu'))(mask_model)

  mask_model = (Conv2DTranspose(64, 3, padding='same', strides=2))(mask_model)
  mask_model = (Conv2D(64, 3, padding='same', activation='relu'))(mask_model)
  mask_model = (Conv2D(64, 3, padding='same', activation='relu'))(mask_model)
  mask_model = (Conv2D(1, 3, padding='same', activation='sigmoid'))(mask_model)


  mask_model = Model(mask_model_in, mask_model)
  return mask_model

mask_model = mask_model_func()

def missing_model_func():
  missing_model_in = Input(shape=(256, 256, 4))

  missing_model = (Conv2D(64, 3, padding='same', activation='relu'))(missing_model_in)
  missing_model = (Conv2D(64, 3, padding='same', activation='relu'))(missing_model)
  missing_model = (Conv2D(64, 3, padding='same', strides=2))(missing_model)

  missing_model = (Conv2D(128, 3, padding='same', activation='relu'))(missing_model)
  missing_model = (Conv2D(128, 3, padding='same', activation='relu'))(missing_model)
  missing_model = (Conv2D(128, 3, padding='same', strides=2))(missing_model)

  missing_model = (Conv2D(256, 3, padding='same', activation='relu'))(missing_model)
  missing_model = (Conv2D(256, 3, padding='same', activation='relu'))(missing_model)
  missing_model = (Conv2D(256, 3, padding='same', strides=2))(missing_model)

  missing_model = (Conv2D(512, 3, padding='same', activation='relu'))(missing_model)
  missing_model = (Conv2D(512, 3, padding='same', activation='relu'))(missing_model)
  missing_model = (Conv2D(512, 3, padding='same', activation='relu'))(missing_model)



  missing_model = (Conv2DTranspose(256, 3, padding='same', strides=2))(missing_model)
  missing_model = (Conv2D(256, 3, padding='same', activation='relu'))(missing_model)
  missing_model = (Conv2D(256, 3, padding='same', activation='relu'))(missing_model)

  missing_model = (Conv2DTranspose(128, 3, padding='same', strides=2))(missing_model)
  missing_model = (Conv2D(128, 3, padding='same', activation='relu'))(missing_model)
  missing_model = (Conv2D(128, 3, padding='same', activation='relu'))(missing_model)

  missing_model = (Conv2DTranspose(64, 3, padding='same', strides=2))(missing_model)
  missing_model = (Conv2D(64, 3, padding='same', activation='relu'))(missing_model)
  missing_model = (Conv2D(64, 3, padding='same', activation='relu'))(missing_model)
  missing_model = (Conv2D(3, 3, padding='same', activation='tanh'))(missing_model)


  missing_model = Model(missing_model_in, missing_model)

  return missing_model

missing_model = missing_model_func()

def create_RFBS():
  in1 = Input(shape=(64, 64, 256))

  block1 = Conv2D(64, (1, 1), padding='same')(in1)
  block1 = Conv2D(64, (3, 3), dilation_rate=1, padding='same')(block1)

  block2 = Conv2D(64, (1, 1), padding='same')(in1)
  block2 = Conv2D(64, (1, 3), padding='same')(block2)
  block2 = Conv2D(64, (3, 3), dilation_rate=3, padding='same')(block2)

  block3 = Conv2D(64, (1, 1), padding='same')(in1)
  block3 = Conv2D(64, (3, 1), padding='same')(block3)
  block3 = Conv2D(64, (3, 3), dilation_rate=3, padding='same')(block3)

  block4 = Conv2D(64, (1, 1), padding='same')(in1)
  block4 = Conv2D(64, (3, 3), padding='same')(block4)
  block4 = Conv2D(64, (3, 3), dilation_rate=5, padding='same')(block4)

  model = Concatenate()([block1, block2, block3, block4])
  model = Conv2D(256, (1, 1))(model)

  model = Add()([in1, model])
  model = Activation('relu')(model)

  model = Model(inputs=in1, outputs=model)
#  model.summary()
  return model

def refiner_model_func():
  refiner_model_in = Input(shape=(256, 256, 3))
  
  refiner_model = Conv2D(64, (7, 7), padding='same')(refiner_model_in)
  refiner_model = Activation('relu')(refiner_model)
  
  refiner_model = Conv2D(128, (3, 3), strides=2, padding='same')(refiner_model)
  refiner_model = Activation('relu')(refiner_model)
  
  refiner_model = Conv2D(256, (3, 3), strides=2, padding='same')(refiner_model)
  refiner_model = Activation('relu')(refiner_model)

  RFBS_block1 = create_RFBS()(refiner_model)
  RFBS_block2 = create_RFBS()(RFBS_block1)
  RFBS_block3 = create_RFBS()(RFBS_block2)
  RFBS_block4 = create_RFBS()(RFBS_block3)
  RFBS_block5 = create_RFBS()(RFBS_block4)
  RFBS_block6 = create_RFBS()(RFBS_block5)
  RFBS_block7 = create_RFBS()(RFBS_block6)
  RFBS_block8 = create_RFBS()(RFBS_block7)
  RFBS_block9 = create_RFBS()(RFBS_block8)

  refiner_model = Conv2DTranspose(128, (3, 3), strides=2, padding='same')(RFBS_block9)
  refiner_model = Activation('relu')(refiner_model)

  refiner_model = Conv2DTranspose(256, (3, 3), strides=2, padding='same')(refiner_model)
  refiner_model = Activation('relu')(refiner_model)

  refiner_model = Conv2D(3, (7, 7), padding='same')(refiner_model)

  refiner_model = Add()([refiner_model_in, refiner_model])
  refiner_model = Activation('tanh')(refiner_model)
  refiner_model = Model(inputs=refiner_model_in, outputs=refiner_model)
  return refiner_model

refiner_model = refiner_model_func()

#Main Model
in_layer = Input(shape=(256, 256, 3))
mask_output = mask_model(in_layer)
missing_in = Concatenate()([in_layer, mask_output])
missing_out = missing_model(missing_in)
mask_output = Reshape((256, 256))(mask_output)
stacked_mask = Lambda(lambda x: K.stack((x, x, x), axis=3), name='stacked_mask')(mask_output)
final = Lambda(lambda x: x[0]*x[1])([in_layer, stacked_mask])
final = Lambda(lambda x: x[0]+x[1])([final, missing_out])
final = refiner_model(final)
model = Model([in_layer], [mask_output, missing_out, final])
model.summary()

model.compile(loss=['binary_crossentropy', 'mse', 'mse'], loss_weights=[1, 1, 4], optimizer=Adam(lr = .0003, clipnorm=1))

model.fit(train_gen, steps_per_epoch=200, validation_data=val_gen, validation_steps=50, epochs=100)
