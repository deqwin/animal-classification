#encoding=utf-8

import os
import tensorflow as tf

import cv2
import numpy as np
from keras import backend as K
from keras.utils import np_utils

from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"]="1"
cwd = os.getcwd()

# 生成 tfrecords 文件
def create_record():
  pathDir = os.listdir(cwd + "/train")
  pathDir.sort()
  
  writer = tf.python_io.TFRecordWriter("tfrecordsResult/train.tfrecords")
  
  for index, name in enumerate(pathDir):
    class_path = cwd +"/train/"+ name+"/"
    for img_name in os.listdir(class_path):
      img_path = class_path + img_name
      img = Image.open(img_path)
      img = img.resize((224, 224))
      img_raw = img.tobytes() #将图片转化为原生bytes

      if img.mode != 'RGB':
        print("here out")
        print(img_path)
        print(img.mode)
          
      example = tf.train.Example(
        features=tf.train.Features(feature={
          "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
          'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        })
      )

      print(img_path + '__' + img.mode)

      writer.write(example.SerializeToString())

  writer.close()

if __name__ == '__main__':
    data = create_record()