# Training

- [ ] Automatically move the trained model into models/{train run name}
- [ ] Get more training datasets 
- [ ] Figure out how to combine multiple datasets
- [ ] Try different resolutions
- [ ] Automate labeling of a video (to make labeling mainly just checking if it's correct)

# General

- [ ] Fix the frame generator in tracker (probably combine the functions).
- [ ] Fix the ball assigner (currently it's just assigning based on left and right distance it should also take into account y)
- [ ] Change the magic ditcs used for managing data into some more readable format like classes
- [ ] Fix the goalkeepr tracking (currently it's not possible to know to which team is the goalkeeper)
- [ ] Use supervision pacakge instead of all the manual drawing
- [ ] Use UMAP for dimensionality reduction and SigLIP for clustering instead of KMeans

# Features

- [ ] Add homography matrix to represent the field
- [ ] Tracking the passes
- [ ] Detecing fouls