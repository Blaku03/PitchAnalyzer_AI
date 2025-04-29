# Milestone 1

Goal of this milestone would be to have a model that can detect players and the ball in a video. 

With that data we can display ball possession, number of passes and current player with ball.

## Training

- [x] Integrate ML backend for auto labeling
- [x] Automatically move the trained model into models/{train run name}
- [x] Try different resolutions
- [x] Automate labeling of a video (to make labeling mainly just checking if it's correct)
- [ ] Label test photos ~900
- [x] Add mlflow for monitoring the training

## Augmentation

- [ ] Do research about possible augmentations that would be useful for this task [good place to start with this task](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/discussion/360331) and later fill this section with tasks. Consider using (albumentations)[https://github.com/albumentations-team/albumentations] for augmentations.
- [ ] Read more about augmentatinos that YOLO does automaticlly and see if we can adjust it or add more augmentations.

## Player prediction model

- [ ] Use UMAP for dimensionality reduction and SigLIP for clustering instead of KMeans [instruction video](https://youtu.be/aBVGKoNZQUw?si=l8EIqtp8bc44Hj3m&t=1778). It's important to note that UMAP is not a clustering algorithm, but it can be used for dimensionality reduction before applying a clustering algorithm like KMeans. The idea is to first reduce the dimensionality of the data using UMAP, and then apply KMeans on the reduced data.
- [x] Currently we initialize the KMeans based on the first frame, so if the predictions are wrong in the first frame, the KMeans will be wrong for the rest of the video. 

## Goalkeeper and Referee tracking

- [ ] Assign goalkeeper to a team based on a distances to middles of teams [instruction video] (https://www.youtube.com/watch?v=aBVGKoNZQUw&t=2700s).

- [ ] Do we need a third cluster for a referee recognistion? How to distinguish him from a goalkeeper in terms of a color of a shirt?
 
## Processing the predictions

- [x] Fix the frame generator in tracker (probably combine the functions).
- [ ] Fix the ball assigner (currently it's just assigning based on left and right distance it should also take into account y)

## Computer vision drawing

- [x] Use supervision pacakge instead of all the manual drawing (if the package is not good enough then we can always go back to the manual drawing)

## General

- [x] Change the magic ditcs used for managing data into some more readable format like classes

# Milestone 2

Goal of this milestone would be to add model able to detect pitch landmark

# Milestone 3

Goal of this milestone would be to have model able to detect fouls, play, throwing etc.
