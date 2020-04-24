import tensorflow as tf
import numpy as np
import os
import pandas as pd
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import seaborn as sns
import glob

from sklearn.model_selection import train_test_split
from utils import get_image_fullpath
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix



class XrayTF:
    def __init__(self, df, IMAGE_SIZE, BATCH_SIZE):
        self.df = df

        self.IMAGE_SIZE = IMAGE_SIZE
        self.BATCH_SIZE = BATCH_SIZE

        self.INPUT_SHAPE = [None, self.IMAGE_SIZE, self.IMAGE_SIZE, 3]
        self.MODEL_URL = "https://tfhub.dev/google/imagenet/resnet_v2_50/classification/4"

    # ----------------------------------
    # MACHINE LEARNING DATA PRE_PROCESS
    # ----------------------------------

    def prepend_image_full_path(self):
        """
        Create column `Image Index` where it contains the full path of each image identity.
        """
        for i in range(len(self.df)):
            self.df["Image Index"].iloc[i] = get_image_fullpath(
                                                self.df["Image Index"].iloc[i], 
                                                self.df.labels.iloc[i]
                                            )

    def generate_image(self, df ,augment=False):
        """
        Batch generate image based on BATCH_SIZE

        @return - Object : batches of images in tensor
        """

        image_generator = self._image_generator(augment)

        df = image_generator.flow_from_dataframe(
            df,
            x_col="Image Index",
            y_col="labels",
            batch_size=self.BATCH_SIZE,
            class_mode="categorical",
            shuffle=True,
            target_size=(self.IMAGE_SIZE, self.IMAGE_SIZE),
        )

        return df

    
    def _image_generator(self, augment):
        """
        Helper for generate_image
        """
        if augment:
            data_augment = tf.keras.preprocessing.image.ImageDataGenerator(
                horizontal_flip = True, 
                height_shift_range = 0.15, 
                width_shift_range = 0.15, 
                rotation_range = 5, 
                shear_range = 0.01,
                fill_mode = 'nearest',
                zoom_range=0.10,
                rescale=1./255
            )
        else:
            data_augment = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255
            )
        
        return data_augment
    
    def get_test_train_split_data(self, NUM_IMAGES=1000):
        """
        Splits the self.df into training and testing data
        """
        self.df = self.df.sample(frac=1)
        train_df, valid_df = train_test_split(
            self.df[:NUM_IMAGES],
            test_size=0.2, 
            random_state=42, 
            stratify = self.df[["labels"]][:NUM_IMAGES]
        )
        return train_df, valid_df


    # Turn prediction probabilities into their respective label (easier to understand)
    def get_label(self, prediction_probabilities):
        """
        Turns an array of prediction probabilities into a label.

        @return - String : the predicted label based on prediction percentage.
        """
        return self.get_unique_labels()[np.argmax(prediction_probabilities)]


    # ----------------------------------
    # MACHINE LEARNING PROCESSING PART       
    # ----------------------------------

    # Build a function to train and return a trained model
    def train_model(self, epochs_num , train_data, val_data):
        """
        Trains a given model and returns the trained version.
        """
        # Create a model
        model = self.create_model()

        # Create new TensorBoard session everytime we train a model
        tensorboard = self.create_tensorboard_callback()
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3) # stops after 3 rounds of no improvements
        # Fit the model to the data passing it the callbacks we created
        model.fit(x=train_data,
                epochs=epochs_num,
                validation_data=val_data,
                verbose=1,
                validation_freq=1, # check validation metrics every epoch
                callbacks=[tensorboard, early_stopping, tf.keras.callbacks.Callback()])

        return model
    
    def create_model(self):
        """
        Helper for train_model Creates a model

        @return - Object : model that is ready to be trained.
        """

        model = tf.keras.Sequential([
            hub.KerasLayer(self.MODEL_URL, input_shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 3)),

            tf.keras.layers.Dense(1024, activation='relu', name='fc1'),
            tf.keras.layers.Dropout(0.5, name='dropout1'),
            tf.keras.layers.Dense(512, activation='relu', name='fc2'),
            tf.keras.layers.Dropout(0.5, name='dropout2'),
            tf.keras.layers.Dense(units=len(self.get_unique_labels()), activation="softmax") # Layer 2 (output layer)
        ])

        # Compile the model
        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy() , # Our model wants to reduce this (how wrong its guesses are)
            optimizer=tf.keras.optimizers.Adam(), # A friend telling our model how to improve its guesses
            metrics=["categorical_accuracy"] # We'd like this to go up
        )

        return model
    
    def model_exist(self, sickness, position):
        """
        Verify if we have a model for a certain disease
        """

        model_folder = os.path.join(Path(__file__).parents[1], "models", f"{sickness}_{position}_*.h5")
        model = glob.glob(model_folder)
        
        if len(model) > 0:
            return self.load_model(model[0])
        return None
    

    # ----------------------------------
    # MACHINE LEARNING POST_PROCESS PART
    # ----------------------------------
    def plot_pred_conf(self, prediction_probabilities, labels, n=1):
        """
        Plots the top 10 highest prediction confidences along with
        the truth label for sample n.
        """
        
        pred_prob, true_label = prediction_probabilities[n], labels[n]

        # Get the predicted label
        true_label = self.get_label(true_label)

        # Find the top 10 prediction confidence indexes
        top_10_pred_indexes = pred_prob.argsort()[-10:][::-1]
        # Find the top 10 prediction confidence values
        top_10_pred_values = pred_prob[top_10_pred_indexes]
        # Find the top 10 prediction labels
        top_10_pred_labels = self.get_unique_labels()[top_10_pred_indexes]

        # Setup plot
        top_plot = plt.bar(np.arange(len(top_10_pred_labels)), 
                            top_10_pred_values, 
                            color="grey")
        plt.xticks(np.arange(len(top_10_pred_labels)),
                    labels=top_10_pred_labels,
                    rotation="vertical")
        
        # Change color of true label
        if np.isin(true_label, top_10_pred_labels):
            top_plot[np.argmax(top_10_pred_labels == true_label)].set_color("green")
        else:
            pass
    
    
    def plot_pred(self,prediction_probabilities, labels, images, n=1):
        """
        View the prediction, ground truth and image for sample n
        """
        pred_prob, true_label, image = prediction_probabilities[n], labels[n], images[n]

        # Get the pred label
        pred_label = self.get_label(pred_prob)
        true_label = self.get_label(true_label)

        # Plot image & remove ticks
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])

        # Change the colour of the title depending on if the prediction is right or wrong
        if pred_label == true_label:
            color = "green"
        else:
            color = "red"
        
        # Change plot title to be predicted, probability of prediction and truth label
        plt.title("{} {:2.0f}% {}".format(pred_label,
                                            np.max(pred_prob)*100,
                                            true_label),
                                            color=color)
    
    def generate_confusion_matrix(self, y_test, y_val):
        cm = pd.DataFrame( confusion_matrix(y_test, y_val, self.get_unique_labels().tolist()) )
        sns.heatmap(cm, annot=True, fmt='d', linewidths=.5)

    def generate_classification_report(self, y_test, y_val):
        print(classification_report(y_test, y_val ))











    # ----------------------------------
    # MACHINE LEARNING UTILS   
    # ----------------------------------
    def get_unique_labels(self):
        return self.df.labels.unique()

    # Create a function to build a TensorBoard callback
    def create_tensorboard_callback(self ):
        # Create a log directory for storing TensorBoard logs
        logdir = os.path.join(Path(__file__).parents[0], "logs")
        return tf.keras.callbacks.TensorBoard(logdir)
    
    # Create a function for viewing images in a data batch
    def show_25_images(self,images, labels):
        """
        Displays 25 images from a data batch.
        """
        # Setup the figure
        plt.figure(figsize=(20, 20))
        # Loop through 25 (for displaying 25 images)
        for i in range(10):
            # Create subplots (5 rows, 5 columns)
            ax = plt.subplot(7, 5, i+1)
            # Display an image
            plt.imshow(images[i])
            # Add the image label as the title
            plt.title(self.get_unique_labels()[labels[i].argmax()])
            # Turn gird lines off
            plt.axis("off")

    def load_model(self, model_path):
        """
        Loads a saved model from a specified path.
        """
        model = tf.keras.models.load_model(model_path,
                                            custom_objects={"KerasLayer":hub.KerasLayer})
        return model

    def save_model(self,model, model_name):
        """
        Saves a given model in a models directory and appends a suffix (str)
        for clarity and reuse.
        """
        model_path = "../models/" + model_name + ".h5" # save format of model
        print(f"Saving model to: {model_path}...")
        model.save(model_path)
    
    def exclude_no_finding(self):
        """
        Remove No Finding label
        """
        self.df = self.df[self.df["labels"] != "No Finding"]
    
    def prune_other_labels(self, array=["Hernia"]):
        """
        Exclude a certain label from a dataframe
        """
        for label in array:
            self.df = self.df[self.df["labels"] != label]

    def prune_based_on_quota(self, quota):
        """
        Exclude a certain label from a dataframe

        @param quota:integer - minimum state count of the labels
        """

        self.df = self.df.groupby("labels").filter(lambda x: len(x) > quota )
    
    def use_one_label(self, label, balance=True):
        """
        Function helper for No finding vs label chosen.
        replaces the dataframe which would now only consist two values.
        """
        self.df = self.df[self.df["labels"].isin([label, "No Finding"])]
        
        if balance:
            df_no_finding = self.df[self.df.labels == "No Finding"]
            df_label = self.df[self.df.labels == label]

            df_no_finding = df_no_finding.sample(len(df_label))
            
            self.df = pd.concat([df_no_finding, df_label], axis=0)

    def balance_all_labels(self):
        """
        takes the minimum count from the value_counts as a reference on how many other labels should have.
        """
        desired_number = self.df.labels.value_counts()[-1]

        df_per_label = [ self.df[self.df.labels == label] for label in self.get_unique_labels()]

        self.df = self.df[0:0]

        for df in df_per_label:
            self.df = pd.concat([self.df, df.sample(desired_number) ], axis=0)
            