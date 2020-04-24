import pandas as pd
import os 
import shutil

from pathlib import Path
from tqdm import tqdm_notebook as tqdm




def get_data_sheet():
    """
    Description
        - load the Data_Entry_2017 csv file
    
    Return
        - DataFrame
    """
    csv_file = os.path.join(Path(__file__).parents[1], "sheet", "Data_Entry_2017.csv")
    df = pd.read_csv(csv_file)
    # fix column names
    df.rename(columns={
            "OriginalImage[Width": "image_width",
            "Height]": "image_width",
            "OriginalImagePixelSpacing[x": "x_pixel_spacing",
            "y]": "y_pixel_spacing",
            "Finding Labels": "labels"
            }, inplace=True)
    df.drop(columns=["Unnamed: 11"], inplace=True)
    
    return df

def get_image_folder():
    """
    Description
        - obtains the filepath of image folder
    
    Return 
        - path name of image folder
    """
    folder = os.path.join(Path(__file__).parents[1], "images")
    return folder
    
def get_all_files_from_image_folder(label):
    """
    Description
        - get all the file names inside images folder.
    
    Param
        label - String - label folder name inside the image folder 

    Return 
        - Array List of all the file names, not absolute path referenced.
    """
    return os.listdir(os.path.join(get_image_folder(), label))

def get_image_fullpath(image_filename, label):
    """
    Description
        - concatinates the image folder with the image filename
    
    Params
        image_filename - string - full file name
        label - string - folder name inside the image folder
    """
    return os.path.join(get_image_folder(), label, image_filename)

def normalize_data_frame(df):
    """
    Description
        - some label data are mixed of sickness IE: sickx|sicky|sickz
        - we extract the data and append another row.
    
    params
        df - pandas DataFrame 

    Return 
        - Cleaned DataFrame
    """
    rows_to_drop = []
    current_index = len(df)

    for i in tqdm(range(len(df))):
        if "|" in df.iloc[i]["labels"]:
            rows_to_drop.append(i)
            labels_to_be_normalized = df.iloc[i]["labels"].split("|")
            
            for label in labels_to_be_normalized:
                data = list(df.iloc[i]) 
                data[1] = label
                df.loc[current_index + 1] = data
                current_index += 1

    df.drop(rows_to_drop, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def create_labels_to_folder(array_of_collection):
    """
    Description
        - on the images folder, it gets too much. We segregate the folder by label

    Params
        array_of_collection - Array[String] - collection of labels
    """

    for label in array_of_collection:

        
        image_label_folder = os.path.join(get_image_folder(), label)

        # create folder if it doesn't exist
        if not os.path.exists(image_label_folder):
            os.mkdir(image_label_folder)

def copy_images_to_label_folder(df):
    
    # for row in df.itertuples(index=True, name='Pandas'):
    #     print(row.c1, row.c2)
    label_folder = os.path.join(get_image_folder(), label)
    file_full_path = os.path.join(get_image_folder(), image_name)

    shutil.copy(file_full_path, label_folder)

        

