# Folder Structure
| clean.py
| train.txt
| test.txt
| SHTpersonRefer_ann.json

# Train/Test Split Rule
There are totally 71 different scenes. 56 training scenes and 15 test scenes. Total 4392 training and 1066 test data. We make sure each word in test set appears at least twice in training set.
```
python clean.py
```