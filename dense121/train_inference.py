from densenet121 import DenseNet
from data_loader import load_data
from keras.optimizers import SGD
from keras.callbacks import Callback, TensorBoard

class LossHistory(Callback):
  def on_train_begin(self, logs={}):
    self.losses = []

  def on_batch_end(self, batch, logs={}):
    loss = logs.get('loss')
    print(loss)
    self.losses.append(loss)

if __name__ == '__main__':
  
  img_rows, img_cols = 224, 224 # Resolution of inputs
  channel = 3
  num_classes = 120
  batch_size = 16
  nb_epoch = 40

  # Load data.
  X_train, Y_train,  X_valid, Y_valid= load_data(img_rows, img_cols)

  # Load our model
  model = DenseNet(classes=num_classes)

  # Start Fine-tuning

  loss = LossHistory()
  model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
  model.fit(X_train, Y_train,
            batch_size=batch_size,
            epochs=nb_epoch,
            shuffle=True,
            verbose=1,
            validation_data=(X_valid, Y_valid),
            callbacks=[loss, TensorBoard(log_dir='./log')]
          )
  model.save_weights('dense121_result.h5')
  # Make predictions
  predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)