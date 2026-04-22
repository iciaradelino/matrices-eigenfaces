# matrices-eigenfaces

an end-to-end eigenfaces project for face representation and recognition using principal component analysis (pca).

the repository contains multiple notebooks that move from a first draft to a more complete and explainable implementation, including:
- manual pca/eigendecomposition
- svd comparison
- reconstruction quality analysis
- nearest-neighbor recognition
- unknown-face thresholding
- custom dataset experiments (classmates vs celebrities)

## project goals

- build an interpretable eigenfaces pipeline from scratch
- compare pca (eigendecomposition route) and svd on the same data
- measure reconstruction error and recognition accuracy as `k` (number of eigenfaces) changes
- test the same pipeline on both benchmark data and a personalized dataset

## notebooks

- `first_draft.ipynb`
  - early baseline using olivetti faces
  - pca + knn classification
  - initial accuracy and reconstruction plots

- `fixed_dataset.ipynb`
  - full, cleaner eigenfaces pipeline on olivetti
  - explicit mean-centering and manual pca math
  - eigenfaces visualization
  - reconstruction vs `k` + mse curve
  - nearest-neighbor recognition in eigenspace
  - unknown-face detection with distance threshold
  - accuracy-vs-`k` sweep + confusion matrix
  - optional `ipywidgets` interactive section

- `personalized_dataset.ipynb`
  - same pipeline on local class/celebrity images
  - uses cropped folders and `128x128` grayscale preprocessing
  - classmate recognition metrics and threshold visualization
  - similarity-style comparisons against celebrity faces

- `pca_vs_svd.ipynb`
  - theoretical and numerical pca-vs-svd comparison
  - sign alignment and component-by-component equivalence checks
  - reconstruction and recognition parity analysis
  - timing benchmark between both approaches on the custom dataset

## data folders

the notebooks reference these local folders:

- `class_pictures/` raw class photos (used by `pca_vs_svd.ipynb`)
- `class_pictures_cropped/` cropped class photos (used by `personalized_dataset.ipynb`)
- `celebrity_pictures/` raw celebrity photos
- `celebrity_pictures_cropped/` cropped celebrity photos (used by `personalized_dataset.ipynb`)

expected organization:

```text
<dataset_root>/
  person_a/
    image1.jpg
    image2.jpg
    ...
  person_b/
    image1.jpg
    ...
```

each subfolder name is treated as a label.

## requirements

install python dependencies:

```bash
pip install numpy matplotlib scikit-learn pillow ipywidgets jupyter
```

> note: no pinned `requirements.txt` is currently included, so the command above installs the needed runtime packages directly.

## quick start

1. create and activate a virtual environment (recommended).
2. install dependencies from the command above.
3. launch jupyter:

```bash
jupyter notebook
```

4. run notebooks in this suggested order:
   1. `fixed_dataset.ipynb`
   2. `personalized_dataset.ipynb`
   3. `pca_vs_svd.ipynb`
   4. `first_draft.ipynb` (for historical reference)

## pipeline summary

the core flow used across notebooks:

1. load images and labels
2. convert to grayscale and resize
3. flatten images into vectors
4. compute and subtract mean face
5. compute eigenfaces (manual pca or svd-based route)
6. project images to eigenspace (`weights`)
7. classify with nearest-neighbor distance
8. optionally reject as unknown if distance exceeds threshold

## key outputs you should see

- top eigenfaces visualized as ghost-like basis faces
- clearer reconstructions as `k` increases
- reconstruction error (mse) decreasing with larger `k`
- recognition accuracy improving with `k` up to a saturation point
- pca and svd producing equivalent components/results (up to sign)

## notes

- olivetti data in `fixed_dataset.ipynb` is fetched with `sklearn.datasets.fetch_olivetti_faces`.
- custom datasets depend on local image folders being present.
- nearest-neighbor distance and threshold choices impact unknown-face behavior.

## supporting docs

- `idea.md`: project scope and target feature set
- `final_structure.md`: notebook/storyboard structure
- `celebrities.md`: candidate celebrity list used for the custom dataset
