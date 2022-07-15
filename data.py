import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, path, list_with_urls, batch_size):
        self.batch_size = batch_size
        self.list_with_urls = np.array(list_with_urls)
        self.len_of_dataset = len(list_with_urls)
        self.path = path

    def on_epoch_end(self):
        np.random.shuffle(self.list_with_urls)

    def __getitem__(self, index):
        start_id = self.batch_size * index
        end_if = min(start_id + self.batch_size, self.len_of_dataset)
        x = [plt.imread(self.path + url) for url in self.list_with_urls[start_id: end_if]]
        x = tf.image.resize(x, (64, 64))
        x = (np.array(x) - 127.5) / 127.5
        return x

    def __len__(self):
        return (self.len_of_dataset // self.batch_size)