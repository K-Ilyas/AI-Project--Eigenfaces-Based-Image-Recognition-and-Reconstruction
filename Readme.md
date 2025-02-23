# Eigenfaces-Based Image Recognition and Reconstruction

## 📌 Project Overview

This project implements **image recognition and reconstruction** using **Principal Component Analysis (PCA)** and the **Eigenfaces method**. The goal is to recognize human faces and reconstruct occluded parts using training data.

### 🔹 Key Features:
- **Compute Eigenfaces** from a dataset of images.
- **Project new images** into the Eigenfaces space.
- **Recognize faces** using **k-Nearest Neighbors (k-NN)**.
- **Reconstruct occluded images** using the closest match from the dataset.
- **Evaluate recognition performance** by analyzing distance distributions.

This project is particularly useful in **face recognition**, **image compression**, and **data dimensionality reduction**.

---


## 🛠 Installation Guide

### 1️⃣ Prerequisites

Before running the project, ensure that you have the following installed:

- **Python 3.7+**
- Required Python libraries:
  
  Install them using:

  ```bash
  pip install numpy matplotlib scikit-learn
  ```

### 2️⃣ Cloning the Repository

To download the source code, run:

```bash
git clone https://github.com/your-username/Eigenfaces-Recognition.git
cd Eigenfaces-Recognition
```

---

## 📜 Implementation Details

### 📌 1. Computing Eigenfaces

```python
def eigenface(X_train, num_components):
    mean_image = np.mean(X_train, axis=0)
    X_centered = X_train - mean_image
    n_samples = X_train.shape[0]
    
    covariance_matrix = np.dot(X_centered, X_centered.T) / n_samples
    eig_vals, eig_vecs = np.linalg.eigh(covariance_matrix)
    
    sorted_indices = np.argsort(eig_vals)[::-1]
    top_indices = sorted_indices[:num_components]
    top_eig_vecs = eig_vecs[:, top_indices]
    
    eigenfaces = np.dot(X_centered.T, top_eig_vecs)
    eigenfaces /= np.linalg.norm(eigenfaces, axis=0)
    
    projections = np.dot(X_centered, eigenfaces)
    
    return eigenfaces, mean_image, projections
```

**Explanation:**
- Computes the **mean face**.
- Centers the training images by subtracting the mean.
- Computes the **covariance matrix** and extracts **eigenvalues** and **eigenvectors**.
- Sorts and selects the **top principal components**.
- Constructs the **Eigenfaces** and projects the training images.

---

### 📌 2. Projecting New Images

```python
def data_project(X, eigenfaces, mean_image):
    X_centered = X - mean_image
    return np.dot(X_centered, eigenfaces)
```

**Explanation:**
- Subtracts the mean face from a given image.
- Projects the image into the **Eigenfaces space**.

---

### 📌 3. Face Recognition Using k-Nearest Neighbors (k-NN)

```python
def recognize_with_kppv(X_test, X_train_projections, k):
    neigh = NearestNeighbors(n_neighbors=k, metric='euclidean')
    neigh.fit(X_train_projections)
    distances, indices = neigh.kneighbors(X_test)
    return indices
```

**Explanation:**
- Uses **k-NN algorithm** to find the **k nearest** faces.
- Returns the **indices of the closest matches**.

---

### 📌 4. Image Reconstruction

```python
def reconstruct_and_visualize(test_image, nearest_image_original, mask_zone):
    test_image = test_image.reshape(400, 300)
    nearest_image_original = nearest_image_original.reshape(400, 300)
    
    reconstructed_image = test_image.copy()
    reconstructed_image[mask_zone] = nearest_image_original[mask_zone]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(test_image, cmap='gray')
    axes[0].set_title('Image de Test (Avec Masque)')
    axes[0].axis('off')
    
    axes[1].imshow(nearest_image_original, cmap='gray')
    axes[1].set_title('Image Non Masquée la Plus Proche')
    axes[1].axis('off')
    
    axes[2].imshow(reconstructed_image, cmap='gray')
    axes[2].set_title('Image Reconstruite')
    axes[2].axis('off')
    
    plt.show()
```

**Explanation:**
- Takes a **masked image** and **nearest match**.
- Replaces the **masked region** with the closest matching region.
- Displays the **original, masked, and reconstructed images**.

---

## 📊 Results and Visualizations

### 🔹 Face Recognition and Reconstruction Example


### 🔹 Distance Distribution for Recognition

---

## 📖 References

- **Principal Component Analysis (PCA)** for dimensionality reduction.
- **Eigenfaces Method** for face recognition.
- **k-Nearest Neighbors (k-NN)** for classification.

---

## 📜 License

This project is licensed under the **MIT License**.

---


