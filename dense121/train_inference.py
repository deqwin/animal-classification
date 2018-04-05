from densenet121 import DenseNet
from data_loader import load_data

if __name__ == '__main__':
  
  img_rows, img_cols = 224, 224 # Resolution of inputs
  channel = 3
  num_classes = 120
  batch_size = 16
  nb_epoch = 40

  # Load Cifar10 data. Please implement your own load_data() module for your own dataset
  X_train, Y_train,  X_valid, Y_valid= load_data(img_rows, img_cols)

  # Load our model
  model = densenet121_model(img_rows=img_rows, img_cols=img_cols, color_type=channel, num_classes=num_classes)
  print(X_train.shape)
  print(Y_train.shape)
  print(X_valid.shape)
  print(Y_valid.shape)
  # Start Fine-tuning

  model.fit(X_train, Y_train,
            batch_size=batch_size,
            nb_epoch=nb_epoch,
            shuffle=True,
            verbose=1,
            validation_data=(X_valid, Y_valid),
          )
  model.save_weights('densenet121_dog120_epoch40_batchsize16_UseSmote_fromSISURFmodelWhile.h5')
  # Make predictions
  predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)