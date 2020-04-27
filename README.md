# GETTING STARTED
- Download file from [Kaggle](https://www.kaggle.com/nih-chest-xrays/data/data)
- create images/ folder from root repo
- create sheet/ folder
- extract and copy folders into the the images, spreadsheets into sheet
- open terminal in images/ folder ```for i in `find . -iname '*.png'`; do mv $i . ; done ```
- delete other folders inside the images

# GETTING STARTED SECOND STEP (OPTIONAL)
- Upload the images frolder within your provider
- Add environment variable IMG_SERVER_URL for the prefix path of your image.

# Environment Variables
- IMG_SERVER_URL : your image hosting server url

# STARTING YOUR RESEARCH ( THIS ONLY WORKS RIGHT NOW ON GOOGLE COLAB SPACE)
- run `!pip uninstall tensorflow`
- run `!pip install tensorflow==2.1`
- run ``` !export IMG_SERVER_URL= {YOUR URL LINK} ```
- restart and run all