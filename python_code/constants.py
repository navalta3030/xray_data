import os

LINK_FOR_YOUR_SERVER = os.getenv(
    "IMG_SERVER_URL", "https://images.marknavalta.com/v0.1/")
HUB_URL = "https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/4"

LOCAL_IMAGES_LOCATION = "/content/images/"

# Main Model
LOSS = "categorical_crossentropy"
ACTIVATION = "elu"
OUTPUT_ACTIVATION = "softmax"
LEARNING_RATE = 0.001
LAYERS_DROPOUT = [
    [[128, 0.5], [64, 0.2]],
]


# Cross validation
CV_LAYERS_DROPOUT = [
    [[256, 0.5]],
    [[128, 0.5], [64, 0.2]],
]
CV_ACTIVATION = ["elu"]
CV_LEARNING_RATE = [0.01, 0.001]
