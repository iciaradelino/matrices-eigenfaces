0. title + intro
   — what eigenfaces are, the core intuition (faces live in a subspace)

1. dataset
   — load olivetti, show sample images, print shapes

2. data preprocessing
   — mean-center explicitly (show the mean face as an image)
   — explain why centering matters before PCA

3. the math: PCA from scratch
   — compute X^T X manually with numpy
   — get eigenvectors/eigenvalues via np.linalg.eigh
   — sort by descending eigenvalue
   — show explained variance curve

4. eigenfaces
   — reshape eigenvectors into images, display top k
   — side note: compare what sklearn PCA gives vs manual

5. face reconstruction
   — reconstruct at k = 5, 20, 50, 100
   — side-by-side originals vs reconstructions
   — plot reconstruction error (MSE) vs k

6. recognition
   — project all faces into eigenspace
   — nearest neighbor by euclidean distance (manual, no sklearn classifier)
   — show: query image → closest match image + distance score

7. unknown face detection
   — add distance threshold
   — test with a non-face image or out-of-distribution image

8. evaluation
   — train/test split, accuracy vs k sweep
   — confusion matrix (optional)

9. custom dataset (students + professors + famous)
   — same pipeline, different data
   — cross-group tests

10. interactive demo (optional)
    — ipywidgets slider for k
    — upload image → see match