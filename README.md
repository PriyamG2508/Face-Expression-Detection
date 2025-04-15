# 😊 Facial Expression Detection

A real-time facial emotion detection web app built with **Streamlit**, **OpenCV**, and a deep learning model trained using **Keras**. This application uses your webcam feed to identify human emotions such as **happy**, **sad**, **angry**, **surprised**, and more.

## 🔍 Features

- Real-time emotion detection via webcam
- Pretrained CNN model for emotion classification
- Detects seven emotions: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`
- Simple, interactive web UI using Streamlit

## 📁 Project Structure

<pre>
📦 facial-expression-detection
├── app.py                  # Streamlit app for real-time detection
├── Trainmodel.ipynb        # Jupyter notebook for model training
├── emotiondetector.h5      # Trained model (see note below)
└── README.md               # Project documentation
</pre>

## 🚀 Getting Started

### Prerequisites

Install the required Python libraries:

```bash
pip install streamlit opencv-python keras tensorflow numpy
```

### Running the App


Place the model file emotiondetector.h5 in the same directory as app.py
(see next section if you don't have it yet)

Launch the Streamlit app:
```bash
streamlit run app.py
```
Allow webcam access and check the "Start Camera" box to begin detecting emotions in real-time.

# 📦 Model File (emotiondetector.h5)

Due to GitHub’s 25MB upload limit, the model is not included in this repository.

You have 2 options:

### Option 1: Download Pretrained Model

🔗 [Google Drive Link to Download Model](https://drive.google.com/file/d/1z-bhICKF3dRKYL9762wsKI5VOteeGii8/view?usp=sharing)
Upload your model to Google Drive and replace this link.

After downloading, place the file in your project root directory.

### Option 2: Train Your Own Model

Use the provided Trainmodel.ipynb notebook. It walks through training a CNN on the [Face recognition dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset). Once training is complete, export the model with:
```bash
model.save("emotiondetector.h5")

```
# 🧪 Model Training Info

The Trainmodel.ipynb notebook:

• Uses CNN architecture suitable for emotion classification

• Accepts grayscale face images resized to 48x48

• Trained on 7 emotion classes using a labeled dataset

You can customize or improve the architecture as needed.

# 📸 Demo

(Will be uploaded soon)

# ✍️ Author
Developed by [Priyam Gupta](https://github.com/PriyamG2508)

Feel free to contribute or open issues for improvements!



