import os
import argparse
from data import DataGenerator
from model import DCGAN


parser = argparse.ArgumentParser(description='')
parser.add_argument('path_with_images', type=str, help='Path with images')
parser.add_argument('path_for_save_model', type=str, help='Path for save model')
args = parser.parse_args()

BATCH_SIZE = 64


def fit(path_with_images, path_for_save_model, batch_size):
    data_generator = DataGenerator(path=path_with_images, list_with_urls=os.listdir(path_with_images), batch_size=100)
    gan = DCGAN(batch_size=batch_size)
    gan.train(data_generator, 5, url_for_save=path_for_save_model)


if __name__ == '__main__':
    fit(args.path_with_images, args.path_for_save_model, BATCH_SIZE)