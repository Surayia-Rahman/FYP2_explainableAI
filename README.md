

# Explainable AI (XAI): Interpreting Convolutional Neural Networks using LIME

This repository contains a full-stack engineering workflow for training deep-learning image classifiers and diagnosing their feature reasoning using a custom-built, from-scratch implementation of **Local Interpretable Model-agnostic Explanations (LIME)**.

The project bridges the gap between deep black-box models, classical machine learning classifiers, and interactive user testing interfaces via native desktop GUIs and web app prototypes.

---

## 🏗️ Repository Architecture

```text
Surayia-Rahman/FYP2_explainableAI/
│
├── notebooks/                        # Development, exploration, and prototyping sandboxes
│   ├── model.ipynb                   # Mass-scale 86-class CNN training architecture
│   ├── Sketch_recognition.ipynb      # Initial bitmap segmentation prototyping 
│   ├── LIME_Exp.ipynb                # Custom LIME exploration on ResNet50 & EfficientNet
│   └── Lime_Original_Explaination.ipynb # Reference baseline utilizing original author library
│
├── src/                              # Production modules and deployment scripts
│   ├── lime_explanation.py           # Core algorithmic class written from scratch
│   ├── main.py                       # Native Tkinter interactive GUI & multi-model harness
│   └── streamlit_application.py      # Streamlit web interface deployment prototype
│
├── assets/                           # Static project media and sample evaluation files
│   └── elephant.jpg                  
│
└── README.md                         # Project documentation

```

---

## 🛠️ Algorithmic Deep-Dive: Our Custom LIME Implementation (`src/lime_explanation.py`)

Rather than relying purely on pre-built wrapper frameworks, this project contains a foundational Python implementation of the LIME algorithm based on the landmark paper:

> *Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier.*

The core `LIME` object wraps any standard `tf.keras.Model` pipeline and handles feature extraction directly in the image space using a mathematical 4-stage loop:

1. **Spatial Segmentation:** Raw pixels carry no intrinsic diagnostic value for an image explanation. The system passes input tensors into a Scikit-Image `quickshift` segmentation routine, clustering raw visual fields into bounded, cohesive **Superpixels** based on local color density and vector boundaries.
2. **Binary Feature Perturbation:** To evaluate what visual regions are critical, a randomized binomial distribution matrix creates $N$ variation vectors. Each vector represents a mask where localized superpixel segments are either left intact (`1`) or turned completely black (`0`), creating a suite of noisy "perturbed" alternative realities of the original file.
3. **Proximity Weighting via Cosine Distance:** The original unaltered image is run through model inference alongside every single masked, perturbed variant. A similarity score is computed between the variation vector and the original target utilizing a Cosine Distance pair metric. Variances that look highly similar to the original image are weighted heavily, while highly altered, unidentifiable image states are penalized.
4. **Training an Interpretable Surrogate Model:** The system fits a lightweight, naturally interpretable machine learning model (such as a `LinearRegression` or `DecisionTreeRegressor`) onto the binary perturbation vectors using the calculated cosine distances as instance sample weights. By pulling the intrinsic attributes of the surrogate model (linear coefficients or structural `feature_importances_`), we extract the exact superpixels that heavily drive or break classification confidence scores.

---

## 🧠 Model Training & Performance (`notebooks/model.ipynb`)

Before interpreting a custom model, a heavy-duty convolutional network was developed from scratch to identify hand-drawn artwork vector boundaries.

* **The Dataset:** Google's open-source **"Quick, Draw!" bitmap corpus**, utilizing a balanced layout across **86 distinct drawing categories** (airplanes, cats, apples, zigzags, etc.) with 5,000 samples per class, totaling **430,000 distinct images**.
* **Preprocessing Pipeline:** Custom vectors are reshaped back into standardized grayscale 4D tensors `(batch, 28, 28, 1)`, normalized to a $[0, 1]$ floating scale, and mapped into automated memory pipelines via `tf.data`.

```text
  [Input Canvas: 28x28x1] ➔ Conv2D (30 filters, 3x3, ReLU) ➔ MaxPooling2D (2x2)
                          ➔ Conv2D (15 filters, 3x3, ReLU) ➔ MaxPooling2D (2x2)
                          ➔ Dropout (0.2) ➔ Flatten
                          ➔ Dense (128, ReLU) ➔ Dense (50, ReLU) ➔ Dense (86, Softmax Output)

```

* **Performance:** Compiled with `categorical_crossentropy` and trained using an `Adam` optimizer, the custom network achieved a **77.88% final multi-class accuracy score** across all 86 drawing targets within 19 epochs.

---

## 🖥️ Application Interfaces & Engineering Insights

This project explores two radically different user interface patterns for real-time model interaction.

### 1. Interactive Multi-Classifier Sandbox (`src/main.py`)

Built natively using **Tkinter**, **OpenCV**, and **Pillow**, this application acts as an on-the-fly custom dataset creator and machine learning laboratory.

* **Dynamic Class Instantiation:** On launch, users name a project and define three distinct drawing targets. The system automatically creates a matching local file directory system.
* **Dataset Generation:** Users free-draw onto a responsive 500x500 canvas using their mouse. Clicking a class button automatically downsamples their sketch into a $50\times50$ pixel thumbnail array and increments it into the proper local data storage directory.
* **On-the-Fly Model Rotation:** Features a model-swapping loop that allows users to instantly train, save, load, and test data across multiple `scikit-learn` algorithms dynamically, including:
* `LinearSVC` (Default)
* `KNeighborsClassifier`
* `LogisticRegression`
* `DecisionTreeClassifier`
* `RandomForestClassifier`
* `GaussianNB`



### 2. Web Interface Prototype (`src/streamlit_application.py`)

This script serves as a conceptual exploration of migrating the local deep-learning inference engine onto a browser environment using **Streamlit**.

> 💡 **Professional Transparency Note on Streamlit Implementation:**
> In this early iteration, the web interface utilizes programmatic coordinate sliders (`st.slider("x")`, `st.slider("y")`) to capture input coordinates and stamp rectangles onto an `ImageDraw` canvas array before sending it to the underlying `cnn_model` engine. While highly effective for isolating deterministic backend testing, it highlights a constraint in Streamlit’s native out-of-the-box UI responsiveness for fluid mouse gesture tracking compared to the Tkinter desktop native canvas.
> **Significance:** This script serves as a critical engineering foundation for state-handling (`st.session_state`) and tensor preprocessing scaling transformations in web environments. It establishes the deployment skeleton required to transition our custom-built CNN and LIME visual explanation pipelines directly to web servers.

---

## 🚀 Getting Started & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Surayia-Rahman/FYP2_explainableAI.git
cd FYP2_explainableAI

```

### 2. Install Dependencies

```bash
pip install tensorflow keras numpy pandas scikit-image scikit-learn matplotlib streamlit opencv-python pillow

```

### 3. Launch the Desktop Tkinter Workspace

To run the free-drawn dataset generator and explore different machine learning estimators:

```bash
python src/main.py

```

### 4. Run the Web Interface Prototype

To launch the browser interface dashboard:

```bash
streamlit run src/streamlit_application.py

```
