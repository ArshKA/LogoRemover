import shutil
from PIL import Image
import numpy as np
import cv2
import random

def load_images(IMAGES_PATH):
  # Loads all images to array
  f = []
  for (dirpath, dirnames, filenames) in walk(IMAGES_PATH):
      f.extend(filenames)
      break
  print(len(f))

  imgs_arr = np.zeros((len(f), 256, 256, 3))

  for i, img_name in enumerate(f):
    if i%(len(f)//10) == 0:
      print('Uploading img {}/{}'.format(i, len(f)))
    img = cv2.imread('IMAGES_PATH/{}'.format(img_name))
    img = cv2.resize(img, (256, 256))
    imgs_arr[i] = img
  return imgs_arr
 
def load_logos(LOGO_PATH):
  # Loads all logos to list
  logo_names = []
  for (dirpath, dirnames, filenames) in walk('/content/logos'):
      logo_names.extend(filenames)
      break
  for img_name in logo_names:
    img = cv2.imread('{}/{}'.format(LOGO_PATH, img_name))
    if np.sum(img) == 0:
      logo_names.remove(img_name)

  random.shuffle(logo_names)
  return logo_names

def combine(orig_img, logo, transparent=.5):
  # Combines image and logo
  # Returns mask of missing section, missing pixels, and completed image
  logo_pad = np.zeros((256, 256, 3))
  y = random.randrange(0, 256-logo.shape[0])
  x = random.randrange(0, 256-logo.shape[1])
  logo_pad[y:y+logo.shape[0], x:x+logo.shape[1]] = logo

  mask = (np.sum(logo_pad, axis=2) <= 0).astype(float)

  zeros = np.array(mask)

  missing = 1-np.stack((zeros, zeros, zeros), axis=2)
  missing = orig_img*missing

  zeros[zeros==0] = transparent
  zeros = np.stack((zeros, zeros, zeros), axis=2)



  img = orig_img*zeros
  img = img+logo_pad
  img = np.clip(img, 0, 255)


  return img, missing, mask

def data_gen(imgs_arr, logo_path, logo_names, batch_size):
  mask_arr = np.zeros((batch_size, 256, 256))
  missing_arr = np.zeros((batch_size, 256, 256, 3))
  logo_img_arr = np.zeros((batch_size, 256, 256, 3))
  clean_img_arr = np.zeros((batch_size, 256, 256, 3))

  while True:
    try:
      for i in range(batch_size):
        r = random.randrange(0, len(imgs_arr))
        clean_img = imgs_arr[r]

        r = random.randrange(0, len(logo_names))
        logo = cv2.imread('{}/{}'.format(logo_path, logo_names[r]))

        logo_img, missing, mask = combine(clean_img, logo, transparent=random.random())

        mask_arr[i], missing_arr[i], logo_img_arr[i] = mask, missing, logo_img
        clean_img_arr[i] = clean_img

      yield [(logo_img_arr-127.5)/127.5], [mask_arr, (missing_arr-127.5)/127.5, (clean_img_arr-127.5)/127.5]
    except: pass
