# House Architecture Style Classification

Image classification of house architectural styles with classical machine learning: hand-crafted
texture and color features, compared across an SVM and a k-NN + Naive Bayes voting ensemble.
Both approaches reach about 81% accuracy on a held-out test set.

## Problem

Given a photo of a house, predict its architectural style. The dataset has two classes, named after
their image folders: `scand` (Scandinavian) and `tudor` (Tudor).

## Dataset

The images are not included in this repository. Both notebooks expect one folder per class:

```
houses/            # training data (split 80/20 into training and validation)
  scand/  *.jpg | *.jpeg | *.png
  tudor/
h_test/            # separate test set
  scand/
  tudor/
```

| Split      | Images | scand | tudor |
|------------|-------:|------:|------:|
| Training   |    839 |     - |     - |
| Validation |    210 |    99 |   111 |
| Test       |    149 |    79 |    70 |

The training/validation split is stratified by class (`random_state=45`). All images are resized to 128x128.

## Feature extraction

Each image is turned into a 517-dimensional feature vector:

- **Texture (5 features):** a gray-level co-occurrence matrix (GLCM) over distances 1 and 2 and four
  directions (0, 45, 90, 135 degrees); contrast, dissimilarity, homogeneity, energy and correlation,
  each averaged over all distances and directions (`scikit-image`).
- **Color (512 features):** a normalized 8x8x8 histogram of the B, G and R channels (`OpenCV`).

The features are standardized with `StandardScaler` fitted on the training set. The ensemble notebook
also rescales them to [0, 1] with `MinMaxScaler`, because Multinomial Naive Bayes needs non-negative inputs.

## Models

| Notebook | Model |
|----------|-------|
| [`svm_classifier.ipynb`](svm_classifier.ipynb) | `SVC(kernel='poly', C=1250, gamma='scale')` |
| [`knn_naive_bayes_ensemble.ipynb`](knn_naive_bayes_ensemble.ipynb) | Hard-voting `VotingClassifier` of k-NN (`k=9`, Manhattan distance), `MultinomialNB(alpha=1.5)` and `BernoulliNB(alpha=1.5)` |

Before settling on the ensemble, k-NN and the Gaussian, Multinomial and Bernoulli Naive Bayes
variants were each evaluated on their own.

## Results

These numbers come from the saved notebook outputs.

| Model | Train acc. | Validation acc. | Test acc. | Test F1 (scand / tudor) |
|-------|-----------:|----------------:|----------:|------------------------:|
| SVM (polynomial kernel) | 99.52% | 80.48% | **81.21%** | 0.81 / 0.82 |
| k-NN + Naive Bayes voting ensemble | 83.67% | **81.43%** | 80.54% | 0.80 / 0.81 |

Both models perform about the same on unseen data. The SVM fits the training set almost perfectly
(99.5%), so it overfits much more than the ensemble, which keeps training and test accuracy within
about 3 points of each other. Both models favour `tudor`, with a test recall of 0.89 (SVM) and 0.87
(ensemble) against 0.75 for `scand`.

Each notebook also includes confusion matrices, a learning curve (5-fold CV), per-class
classification reports, training and inference timings, and a feature-distribution box plot. It can
also export every validation and test image, with the true and predicted labels in the file name.

## How to run

The notebooks were developed in Google Colab with the dataset on Google Drive.

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Edit `train_folder` and `test_folder` in the data-loading cell so they point to your copy of the
   dataset. Also edit `save_dir` in the last two cells if you want to export the classified images.
3. Start Jupyter and run all cells:
   ```bash
   jupyter notebook svm_classifier.ipynb
   ```

In Colab, mount Google Drive first (`from google.colab import drive; drive.mount('/content/drive')`).

## Tech stack

Python, scikit-learn, scikit-image, OpenCV, NumPy, Matplotlib, Jupyter / Google Colab.

## Project structure

```
.
├── svm_classifier.ipynb             # GLCM + color histogram features, polynomial-kernel SVM
├── knn_naive_bayes_ensemble.ipynb   # Same features, k-NN + Naive Bayes voting ensemble
├── requirements.txt
└── README.md
```

## Author

George-Armand Ursei - [GitHub](https://github.com/armandursei)
