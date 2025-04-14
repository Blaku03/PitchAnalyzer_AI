# Milestone 1

Goal of this milestone would be to have a model that can detect players and the ball in a video. 

With that data we can display ball possession, number of passes and current player with ball.

## Training

- [x] Integrate ML backend for auto labeling
- [x] Automatically move the trained model into models/{train run name}
- [x] Try different resolutions
- [x] Automate labeling of a video (to make labeling mainly just checking if it's correct)
- [ ] Label test photos ~900
- [ ] Add mlflow for monitoring the training

## Augmentation

- [ ] Do research about possible augmentations that would be useful for this task [good place to start with this task](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/discussion/360331) and later fill this section with tasks


## Player prediction model

- [ ] Use UMAP for dimensionality reduction and SigLIP for clustering instead of KMeans [instruction video](https://youtu.be/aBVGKoNZQUw?si=l8EIqtp8bc44Hj3m&t=1778)

## Processing the predictions

- [ ] Fix the frame generator in tracker (probably combine the functions).
- [ ] Fix the ball assigner (currently it's just assigning based on left and right distance it should also take into account y)

## Computer vision drawing

- [ ] Use supervision pacakge instead of all the manual drawing (if the package is not good enough then we can always go back to the manual drawing)

## General

- [ ] Change the magic ditcs used for managing data into some more readable format like classes

# Milestone 2

Goal of this milestone would be to add model able to detect pitch landmark

# Milestone 3

Goal of this milestone would be to have model able to detect fouls, play, throwing etc.
