import os

LINK_FOR_YOUR_SERVER = os.getenv(
    "IMG_SERVER_URL", "")
HUB_URL = "https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/4"

LOCAL_IMAGES_LOCATION = "/content/images/"

# ml model
LOSS = "categorical_crossentropy"
OUTPUT_ACTIVATION = "softmax"
LEARNING_RATE = 0.001
