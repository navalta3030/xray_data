from utils import get_data_sheet, normalize_data_frame
from ml_utils import XrayTF
from tqdm import tqdm_notebook as tqdm
from threading import Thread
from constants import LOCAL_IMAGES_LOCATION

import sys
import urllib.request as req
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import shutil
import time
import argparse


class Downloader(Thread):
    def __init__(self, file_url, save_path):
        Thread.__init__(self)
        self.file_url = file_url
        self.save_path = save_path

    def run(self):
        remaining_download_tries = 15

        while remaining_download_tries > 0:
            try:
                req.urlretrieve(self.file_url, self.save_path)
                time.sleep(0.1)
            except:
                print("error downloading " + self.file_url +
                      " on trial no: " + str(16 - remaining_download_tries))
                remaining_download_tries = remaining_download_tries - 1
                continue
            else:
                break


class SickNessPipeLine(XrayTF):
    def __init__(self, ImageSize, Sickness, BatchSize, CsvFile, Epoch):
        """
        Main purpose of this one is a pipeline for each sickness

        @param - ImageSize : Integer       - pixel size of the image
        @param - Sickness  : String        - What sickness do you want to perform ML
        @param - BatchSize : Integer       - how many batches within the epoch
        @param - CsvFile   : String        - file location of the csv that is to be processed
        @param - Epoch     : Integer       - number of epochs
        """

        self.ImageSize = ImageSize
        self.Sickness = Sickness
        self.BatchSize = BatchSize
        self.CsvFile = CsvFile
        self.Epoch = Epoch

        self.train_df = None
        self.val_df = None
        self.test_df = None

        self.train_data = None
        self.val_data = None
        self.test_data = None

        if os.path.exists(self.CsvFile):
            df = pd.read_csv(self.CsvFile)
            super().__init__(df, self.ImageSize, self.BatchSize)
            self.prepend_image_full_path()
        else:
            df = get_data_sheet()
            df = normalize_data_frame(df)
            super().__init__(df, self.ImageSize, self.BatchSize)
            self.prepend_image_full_path()
            self.df.to_csv(self.CsvFile, index=False)

    def _pre_process(self):
        print("="*50)
        print(self.df.labels.value_counts())

        # Pick a sickness
        self.use_one_label(SICKNESS, balance=True)

        # Verify how balanced our data is
        print("="*50)
        print(self.df.labels.value_counts())

        self.download_images()

        #  let's now append the full system path
        self.df["Image Index"] = self.df["Image Index"].apply(
            lambda x: LOCAL_IMAGES_LOCATION + x.split('/')[-1])

        # Split them into training and validation using NUM_IMAGES
        self.train_df, self.valid_df, self.test_df = self.get_test_train_split_data(
            len(self.df))

        print(self.train_df.labels.value_counts())
        print("="*50)
        print(self.valid_df.labels.value_counts())
        print("="*50)
        print(self.test_df.labels.value_counts())

    def _main_process(self):
        # Create training and validation data batches using ImageDataGenerator
        self.train_data = self.generate_image(self.train_df, True)
        self.val_data = self.generate_image(self.valid_df)
        self.test_data = self.generate_image(self.test_df)

        print("="*50)
        if self.model_exist(self.Sickness):
            model = self.model_exist(self.Sickness)
            a, b = model.evaluate(self.test_data)
        else:
            model, stop = self.train_model(
                self.Epoch, self.train_data, self.val_data)
            a, b = model.evaluate(self.test_data)

            filename = f"{self.Sickness}_{b:.2f}"
            self.save_model(model, filename)
        print("="*50)

        self.visualize_after_training(model, stop)

        # Make predictions on the validation data (not used to train on)
        predictions = model.predict(self.test_data, verbose=1)

        print("="*50)
        print(f"Loss : {a}, Accuracy: {b}")

        # convert [0,25, 0.75] -> "No Finding/Effusion"
        y_test = [self.get_label(x) for x in predictions]
        y_val = [self.get_unique_labels()[x] for x in self.test_data.labels]

        self.generate_confusion_matrix(y_test, y_val)
        self.generate_classification_report(y_test, y_val)

    def visualize_after_training(self, model, stop):
        """
        Visualizes the loss/val_loss and acc/val_acc
        """
        acc = model.history.history['accuracy']
        val_acc = model.history.history['val_accuracy']

        loss = model.history.history['loss']
        val_loss = model.history.history['val_loss']

        epochs_range = range(stop.stopped_epoch + 1)

        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    def download_images(self):
        """
        Downloads the images from the prepended link
        """

        if not os.path.exists(LOCAL_IMAGES_LOCATION):
            os.mkdir(LOCAL_IMAGES_LOCATION)

            self.df["Image Index"].apply(lambda x: Downloader(
                x, f"{LOCAL_IMAGES_LOCATION}{x.split('/')[-1]}").start())

    def run(self):
        self._pre_process()
        self._main_process()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configuration')

    parser.add_argument("-img", "--ImageSize",    type=int,
                        help="Image size to be trained", required=True)

    parser.add_argument("-s",  "--Sickness",   type=str,
                        help="What Sickness should we train on?", required=True)

    parser.add_argument("-b",   "--BatchSize",    type=int,
                        help="Batch size for training", default=32)

    parser.add_argument("-f", "--CsvFile",      type=str,
                        help="This should be located within ../sheet/", required=True)

    parser.add_argument("-e", "--Epoch",      type=int,
                        help="Number of Epochs", default=10)

    args = parser.parse_args()

    SICKNESS = args.Sickness
    BATCH_SIZE = args.BatchSize
    IMAGE_SIZE = args.ImageSize
    CSV_FILE = args.CsvFile
    EPOCH = args.Epoch

    csv_file_full_path = f'/content/xray_code/sheet/{CSV_FILE}'

    pipeline = SickNessPipeLine(
        ImageSize=IMAGE_SIZE,
        Sickness=SICKNESS,
        BatchSize=BATCH_SIZE,
        CsvFile=CSV_FILE,
        Epoch=EPOCH
    )

    pipeline.run()
