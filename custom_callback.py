from tensorflow import keras

class CustomCallback(keras.callbacks.Callback):
    def __init__(self, model_name, save_best_metric='val_loss', this_max=False):
        self.save_best_metric = save_best_metric
        self.max = this_max
        if this_max:
            self.best = float('-inf')
        else:
            self.best = float('inf')

        self.model_name = model_name
        self.best_epoch = -1

        self.epochs = []
        self.losses = []
        self.val_losses = []


    def save_best(self, metric_value, epoch):
        self.best = metric_value
        self.best_weights = self.model.get_weights()
        self.best_epoch = epoch
        with open(self.model_name + '_best_epoch.txt', 'a') as f:
            f.write(str(self.best_epoch) + ',' + str(metric_value) + '\n')


    def on_epoch_end(self, epoch, logs=None):
        metric_value = logs[self.save_best_metric]
        if (
            (self.max and metric_value > self.best) or
            (not self.max and metric_value < self.best)
        ):
            self.save_best(metric_value, epoch)

        if epoch > 1:
            self.epochs.append(epoch)
            self.losses.append(logs['loss'])
            self.val_losses.append(logs['val_loss'])