import os
import numpy as np
import keras.backend as K
from keras.datasets import mnist
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from CONSTANTS import *
from controller import Controller
from mlp_generator import MLPGenerator


class MLPNAS(Controller):

    def __init__(self, x, y):

        self.x = x
        self.y = y
        self.target_classes = TARGET_CLASSES
        self.controller_sampling_epochs = CONTROLLER_SAMPLING_EPOCHS
        self.samples_per_controller_epoch = SAMPLES_PER_CONTROLLER_EPOCH
        self.controller_train_epochs = CONTROLLER_TRAINING_EPOCHS
        self.architecture_train_epochs = ARCHITECTURE_TRAINING_EPOCHS
        self.controller_loss_alpha = CONTROLLER_LOSS_ALPHA

        self.data = []


        super().__init__()

        self.model_generator = MLPGenerator()

        self.controller_batch_size = len(self.data)
        self.controller_input_shape = (1, MAX_ARCHITECTURE_LENGTH - 1)
        self.controller_model = self.hybrid_control_model(self.controller_input_shape, self.controller_batch_size)

    def unison_shuffled_copies(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def create_architecture(self, sequence):
        if self.target_classes == 2:
            self.model_generator.loss_func = 'binary_crossentropy'
        model = self.model_generator.create_model(sequence, np.shape(self.x[0]))
        model = self.model_generator.compile_model(model)
        return model

    def train_architecture(self, model):
        x, y = self.unison_shuffled_copies(self.x, self.y)
        history = self.model_generator.train_model(model, x, y, self.architecture_train_epochs)
        return history

    def append_model_metrics(self, sequence, history, pred_accuracy):
        if len(history.history['val_accuracy']) == 1:
            self.data.append([sequence,
                              history.history['val_accuracy'][0],
                              pred_accuracy])
        else:
            self.data.append([sequence,
                              np.ma.average(history.history['val_accuracy'],
                                            weights=np.arange(1, len(history.history['val_accuracy']) + 1),
                                            axis=-1),
                              pred_accuracy])

    def prepare_controller_data(self, sequences):
        controller_sequences = pad_sequences(sequences, maxlen=self.max_len, padding='post')
        xc = controller_sequences[:, :-1].reshape(len(controller_sequences), 1, self.max_len - 1)
        yc = to_categorical(controller_sequences[:, -1], self.controller_classes)
        val_acc_target = [item[1] for item in self.data]
        return xc, yc, val_acc_target

    def get_discounted_reward(self, rewards):
        discounted_r = np.zeros_like(rewards, dtype=np.float32)
        running_add = 0.
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.controller_loss_alpha + rewards[t]
            discounted_r[t] = running_add
        discounted_r -= discounted_r.mean() / discounted_r.std()
        return discounted_r

    def custom_loss(self, target, output):
        baseline = 0.5
        reward = np.array([item[1] - baseline for item in self.data[-self.samples_per_controller_epoch:]]).reshape(
            self.samples_per_controller_epoch, 1)
        discounted_reward = self.get_discounted_reward(reward)
        loss = - K.sum(target * K.log(output / K.sum(output)), axis=-1) * discounted_reward
        return loss

    def sort_search_data(self):
        val_accs = [item[1] for item in self.data]
        sorted_idx = np.argsort(val_accs)[::-1]
        self.data = [self.data[x] for x in sorted_idx]

    def search(self):
        for controller_epoch in range(self.controller_train_epochs):
            print('-----------------Controller Epoch: {}-----------------'.format(controller_epoch))
            sequences = self.sample_architecture_sequences(self.controller_model, self.samples_per_controller_epoch)
            pred_accuracies = self.get_predicted_accuracies_hybrid_model(self.controller_model, sequences)
            for i, sequence in enumerate(sequences):
                model = self.create_architecture(sequence)
                history = self.train_architecture(model)
                self.append_model_metrics(sequence, history, pred_accuracies[i])
            xc, yc, val_acc_target = self.prepare_controller_data(sequences)
            self.train_hybrid_model(self.controller_model,
                                    xc,
                                    yc,
                                    val_acc_target[-self.samples_per_controller_epoch:],
                                    self.custom_loss,
                                    len(self.data),
                                    self.controller_train_epochs)
        self.sort_search_data()
        return self.data


if __name__ == '__main__':
    (x, y), (_, _) = mnist.load_data()
    nas_object = MLPNAS(x[:100], y[:100])
    data = nas_object.search()
    for x in data[:TOP_N]:
        print(nas_object.decode_sequence(x[0]), "--> val acc:", x[1], "predicted val acc:", x[2])