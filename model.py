import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class DCGAN:
    def __init__(self, batch_size=100, optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)):
        self.discriminator = None
        self.generator = None

        self.init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        self.binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.batch_size = batch_size
        self.optimizer = optimizer

        self.build_discriminator()
        self.build_generator()

        self.gen_losses = []
        self.disc_losses = []

    def build_generator(self):
        self.generator = tf.keras.models.Sequential()

        self.generator.add(tf.keras.layers.Input(shape=(1, 1, 100)))

        self.generator.add(
            tf.keras.layers.Conv2DTranspose(filters=1024, strides=(1, 1), kernel_size=(4, 4), padding='valid',
                                            use_bias=False, kernel_initializer=self.init))
        self.generator.add(tf.keras.layers.BatchNormalization())
        self.generator.add(tf.keras.layers.LeakyReLU(0.2))

        self.generator.add(
            tf.keras.layers.Conv2DTranspose(filters=512, strides=(2, 2), kernel_size=(4, 4), padding='same',
                                            use_bias=False, kernel_initializer=self.init))
        self.generator.add(tf.keras.layers.BatchNormalization())
        self.generator.add(tf.keras.layers.LeakyReLU(0.2))

        self.generator.add(
            tf.keras.layers.Conv2DTranspose(filters=256, strides=(2, 2), kernel_size=(4, 4), padding='same',
                                            use_bias=False, kernel_initializer=self.init))
        self.generator.add(tf.keras.layers.BatchNormalization())
        self.generator.add(tf.keras.layers.LeakyReLU(0.2))

        self.generator.add(
            tf.keras.layers.Conv2DTranspose(filters=128, strides=(2, 2), kernel_size=(4, 4), padding='same',
                                            use_bias=False, kernel_initializer=self.init))
        self.generator.add(tf.keras.layers.BatchNormalization())
        self.generator.add(tf.keras.layers.LeakyReLU(0.2))

        self.generator.add(
            tf.keras.layers.Conv2DTranspose(filters=3, strides=(2, 2), kernel_size=(4, 4), padding='same',
                                            use_bias=False, kernel_initializer=self.init))
        self.generator.add(tf.keras.layers.BatchNormalization())
        self.generator.add(tf.keras.layers.Activation('tanh'))

    def build_discriminator(self):
        self.discriminator = tf.keras.models.Sequential()

        self.discriminator.add(tf.keras.layers.Input(shape=(64, 64, 3)))

        self.discriminator.add(
            tf.keras.layers.Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same', use_bias=False,
                                   kernel_initializer=self.init))
        self.discriminator.add(tf.keras.layers.BatchNormalization())
        self.discriminator.add(tf.keras.layers.LeakyReLU(0.2))

        self.discriminator.add(
            tf.keras.layers.Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', use_bias=False,
                                   kernel_initializer=self.init))
        self.discriminator.add(tf.keras.layers.BatchNormalization())
        self.discriminator.add(tf.keras.layers.LeakyReLU(0.2))

        self.discriminator.add(
            tf.keras.layers.Conv2D(filters=512, kernel_size=(4, 4), strides=(2, 2), padding='same', use_bias=False,
                                   kernel_initializer=self.init))
        self.discriminator.add(tf.keras.layers.BatchNormalization())
        self.discriminator.add(tf.keras.layers.LeakyReLU(0.2))

        self.discriminator.add(
            tf.keras.layers.Conv2D(filters=1024, kernel_size=(4, 4), strides=(2, 2), padding='same', use_bias=False,
                                   kernel_initializer=self.init))
        self.discriminator.add(tf.keras.layers.BatchNormalization())
        self.discriminator.add(tf.keras.layers.LeakyReLU(0.2))

        self.discriminator.add(
            tf.keras.layers.Conv2D(filters=1, kernel_size=(4, 4), strides=(1, 1), padding='valid', use_bias=False,
                                   kernel_initializer=self.init))
        self.discriminator.add(tf.keras.layers.BatchNormalization())

        self.discriminator.add(tf.keras.layers.Flatten())

    def discriminator_loss(self, real_pred, fake_pred):
        real_y = np.ones((real_pred.shape))
        fake_y = np.zeros((fake_pred.shape))

        real_loss = self.binary_crossentropy(real_y, real_pred)
        fake_loss = self.binary_crossentropy(fake_y, fake_pred)

        return real_loss + fake_loss

    def generator_loss(self, fake_pred):
        fake_y = np.ones((fake_pred.shape))
        fake_loss = self.binary_crossentropy(fake_y, fake_pred)

        return fake_loss

    @tf.function
    def train_step(self, real_x):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            noise = np.random.uniform(-2, 2, size=(self.batch_size, 1, 1, 100))
            fake_x = self.generator(noise)

            fake_pred = self.discriminator(fake_x)
            real_pred = self.discriminator(real_x)

            gen_loss = self.generator_loss(fake_pred)
            disc_loss = self.discriminator_loss(real_pred, fake_pred)

        gen_grad = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_grad = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.optimizer.apply_gradients(zip(gen_grad, self.generator.trainable_variables))
        self.optimizer.apply_gradients(zip(disc_grad, self.discriminator.trainable_variables))

        return gen_loss, disc_loss, fake_x

    def train(self, dataset, epochs, url_for_save):
        for epoch in range(epochs):
            batch_id = 0
            for batch in dataset:
                gen_loss, disc_loss, preds = self.train_step(batch)

                self.gen_losses.append(gen_loss)
                self.disc_losses.append(disc_loss)

                if batch_id % 10 == 0:
                    self.generator.save(url_for_save + 'DCGAN_generatorV2.hdf5')
                    self.generator.save(url_for_save + 'DCGAN_discriminatorV2.hdf5')


                    print(f'{epoch} / {epochs} эпох', end='\t\t')
                    print(f'{batch_id} / {len(dataset)} батчей')

                    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(12, 8))
                    for i in range(5):
                        axes[i].imshow(preds[i])
                    plt.show()

                batch_id += 1
