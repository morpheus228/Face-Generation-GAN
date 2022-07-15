import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from model import DCGAN

gan = DCGAN(batch_size=100)
gan.discriminator = tf.keras.models.load_model('models/DCGAN_discriminatorV2.hdf5')
gan.generator = tf.keras.models.load_model('models/DCGAN_generatorV2.hdf5')


images = gan.generator(np.random.normal(size=[20, 1, 1, 100]))
for i in images:
    plt.imshow(i)
    plt.show()