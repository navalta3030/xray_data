# GETTING STARTED
- Download file from [Kaggle](https://www.kaggle.com/nih-chest-xrays/data/data)
- create images/ folder from root repo
- create sheet/ folder
- extract and copy folders into the the images, spreadsheets into sheet
- for i in `find . -iname '*.jpg'`; do mv $i . ; done
- delete other folders inside the images