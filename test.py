from data.DataProcessing import *

import keras.backend as K
from keras.models import Model, Sequential, load_model
import keras
from keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, Dense, Input, Reshape, Lambda, Dropout, Concatenate, Add, Activation
from keras.optimizers import Adam

IMAGES_PATH = ''
LOGO_PATH = ''

imgs_arr = load_images(IMAGES_PATH)
logo_names = load_logos(LOGO_PATH)

model = load_model('imageFiller.keras')

def test_data_gen(imgs_arr, logo_path, logo_names):
  mask_arr = np.zeros((256, 256))
  missing_arr = np.zeros((256, 256, 3))
  logo_img_arr = np.zeros((256, 256, 3))
  clean_img_arr = np.zeros((256, 256, 3))
  while True:
    try:
      r = random.randrange(0, len(imgs_arr))
      clean_img = imgs_arr[r]

      r = random.randrange(0, len(logo_names))
      logo = cv2.imread('{}/{}'.format(logo_path, logo_names[r]))

      logo_img, missing, mask = combine(clean_img, logo, transparent=random.random())

      mask_arr, missing_arr, logo_img_arr = mask, missing, logo_img
      clean_img_arr = clean_img

      return [(logo_img_arr-127.5)/127.5], [mask_arr, (missing_arr-127.5)/127.5, (clean_img_arr-127.5)/127.5]
    except: pass
    
a = test_data_gen(imgs_arr[:len(imgs_arr)//5], LOGO_PATH, logo_names[:len(logo_names)//5])
cv2.imwrite('output/logo.png', a[0][0]*127.5+127.5)
cv2.imwrite('output/missing.png', a[1][1]*127.5+127.5)
cv2.imwrite('output/correct.png', a[1][2]*127.5+127.5)

pred = model.predict(np.array(a[0]))
cv2.imwrite('output/pred_mask.png', pred[0][0]*255)
cv2.imwrite('output/pred_missing.png', pred[1][0]*127.5+127.5)
cv2.imwrite('output/pred_correct.png', pred[2][0]*127.5+127.5)
