# Vehicle Recognition (Bike vs Car)

This project is an end-to-end **Image Classification application** that classifies images as **Bike** or **Car** using **Machine Learning** and deploys the model using **Streamlit**.

---

## 📌 Project Overview

- Built an image classification model to distinguish between **Bike** and **Car** images.
- Performed proper **image preprocessing** to standardize the dataset.
- Trained a **Logistic Regression** model as a baseline ML approach.
- Achieved **~95% test accuracy** after refining the problem into binary classification.
- Deployed the trained model using **Streamlit** for real-time image prediction.

---

## 🗂️ Dataset Structure

Dataset (Vehicles)/
├── bike/
│ └── bike images (.jpg)
├── car/
│ └── car images (.jpg)


Each folder represents one class.

---

## 🔧 Image Preprocessing

The following preprocessing steps were applied:
- Resized all images to **224 × 224**
- Converted images to **RGB format**
- Normalized pixel values
- Flattened images for model training

---

## 🤖 Model Used

- **Algorithm:** Logistic Regression  
- **Why Logistic Regression?**
  - Simple and interpretable baseline model
  - Helps understand data behavior before moving to deep learning
- **Performance:**
  - Training Accuracy: **100%**
  - Test Accuracy: **~95%**

---

## 🌐 Deployment

The trained model was deployed using **Streamlit**:
- Upload an image (`.jpg / .png`)
- Model predicts whether the image is a **Bike** or a **Car**
- Simple and interactive web interface

---

## 🛠️ Tech Stack

- Python
- NumPy
- Pillow (PIL)
- Scikit-learn
- Streamlit

---

## 🚀 How to Run the Project Locally

### 1️⃣ Clone the repository
```bash
git clone <your-repo-link>
cd Vehical_Recognition

