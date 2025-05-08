# 🏟️ PitchAnalyzer_AI
By [Blaku03](https://github.com/Blaku03) and [gruzewson](https://github.com/gruzewson)

## 🧭 Overview

**PitchAnalyzer_AI** is a sports analytics framework designed to turn raw football match video into actionable insights. At its core, the system:

- **Detects** players and the ball in each frame.  
- **Assigns** each player to their team.  
- **Tracks** players and ball trajectories over time.  
- **Maps** camera views to a bird’s‑eye perspective.  
- **Aggregates** events into game statistics.

If you are insterested how we created this project, what challenges we faced, and how we solved them, check out our [Technical Documentation](technical_documentation.md).

## Table of Contents

-  [🎦Demo](#demo)
-  [👀Try it out!](#try-it-out)
-  [⚙️Project workflow](#project-workflow)
-  [🔜Future plans](#future-plans)

## <a name="demo"></a> 🎦 Demo

Here’s a quick demo of PitchAnalyzer_AI in action:

![Demo of PitchAnalyzer_AI](documentation_resources/Demo_Pitch.gif)

## <a name="try-it-out"></a> 👀 Try it out!

Explore the project in action with interactive Jupyter Notebooks!

The notebooks are designed to run online on Kaggle or Google Colab as well locally on your machine.

### 📌 Featured Notebooks 
* **Game Annotator** [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Blaku03/PitchAnalyzer_AI/blob/main/development_and_analysis/game_annotator.ipynb) [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/blaku03/gameannotator) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/Blaku03/PitchAnalyzer_AI/blob/main/development_and_analysis/game_annotator.ipynb)

* **Pitch Mapping** [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Blaku03/PitchAnalyzer_AI/blob/main/development_and_analysis/pitch_mapping.ipynb) [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/blaku03/pitchmapping) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/Blaku03/PitchAnalyzer_AI/blob/main/development_and_analysis/pitch_mapping.ipynb)

* **Model Training** (we trained our models mainly on kaggle) [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/blaku03/gamemodeltraining) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/Blaku03/PitchAnalyzer_AI/blob/main/development_and_analysis/game_annotator.ipynb)

### Resources

* Our models can be found on kaggle [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/models/blaku03/player-detection/) as well as our labeled dataset [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/datasets/blaku03/pitchannotatoins/data)

## <a name="project-workflow"></a> ⚙️ Project workflow

![project workflow diagram](documentation_resources/Workflow.png)

1. **Raw Data & Events**  
   - 30 s match clips + manually logged events as inputs.

2. **Auto‑Labeling (Label Studio)**  
   - Semi‑automated generation of player/ball bounding boxes and pitch keypoints, with model‑assisted corrections.

3. **Model Training**  
   - YOLOv11 for player/ball detection  
   - Keypoint detector + homography for field mapping  
   - Play‑recognition module combining detections and geometry

4. **Post‑Processing & Analytics**  
   - Derive ball possession, player heatmaps, bird’s‑eye view, distances covered, and other stats from model outputs.

## <a name="future-plans"></a> 🔜Future plans

When we have more time, we plan to:

- Develop a model for detecting specific plays and game strategies.
- Improve the accuracy and robustness of our current models for player and ball detection.
- Expand the range of statistics generated, such as advanced metrics for player performance and team dynamics.
- Implement a more sophisticated method for recognizing teams, potentially replacing KNN with a deep learning-based approach for better accuracy and scalability.