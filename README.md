✍️ Handwritten Alphabet Classification Using CNN | TensorFlow & Keras

Built and evaluated deep learning models to classify handwritten English alphabet characters using Convolutional Neural Networks (CNNs). This project focuses on end-to-end image classification, from data preprocessing to model training, evaluation, and performance analysis.

The model was trained on a large-scale handwritten alphabet dataset consisting of grayscale 28×28 pixel images representing 26 English letters. Multiple neural network architectures were implemented and improved to optimize classification accuracy and generalization.

This project demonstrates hands-on experience with deep learning workflows, CNN architecture design, and model evaluation using TensorFlow and Keras. 

1beabf42-5813-4dc7-b87f-28dfb93…

🧠 Problem Statement

Automatically recognize handwritten English alphabet characters by learning spatial features from pixel-level image data using deep neural networks.

🛠 Tools & Technologies

Python

TensorFlow

Keras

Convolutional Neural Networks (CNN)

NumPy & Pandas

Matplotlib & Seaborn

Jupyter Notebook

📊 Dataset

Source: Kaggle – AZ Handwritten Alphabets

Size: ~370,000 grayscale images

Image Shape: 28 × 28 pixels

Classes: 26 (A–Z)

Split: 80% Training / 20% Testing

🔍 Key Project Steps
1️⃣ Data Preprocessing

Loaded and normalized grayscale image data (pixel values scaled to 0–1)

Reshaped input data for CNN compatibility

Performed train/test split with validation subset

2️⃣ Model Design & Implementation

Built CNN architectures with:

Convolutional layers for feature extraction

Pooling layers for dimensionality reduction

Dense layers for classification

Softmax output layer for multi-class prediction

Used categorical cross-entropy loss and Adam optimizer

3️⃣ Model Training

Trained models with validation monitoring

Tuned hyperparameters such as epochs, batch size, and learning rate

Applied callbacks (EarlyStopping, ModelCheckpoint) to prevent overfitting

4️⃣ Model Evaluation

Evaluated performance using accuracy and loss metrics

Generated classification reports and confusion matrices

Analyzed per-class performance across all alphabet characters

5️⃣ Model Improvements

Enhanced performance using regularization techniques (Dropout)

Adjusted CNN depth and filter sizes for better feature learning

📈 Results & Insights

Achieved high classification accuracy on unseen test data

CNN models significantly outperformed baseline dense networks

Certain visually similar characters showed higher confusion, highlighting real-world OCR challenges


