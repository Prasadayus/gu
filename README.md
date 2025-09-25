# Signature Verification Project Documentation

This document outlines the process of creating a signature verification system. It covers dataset creation, model selection, the training process, and a comprehensive evaluation of a real vs. fake signature classification pipeline. The system can detect and classify signatures from unseen documents provided via a URL.

## 1. Dataset Creation

A robust and diverse dataset is crucial for training an effective signature verification model. This project utilized a combination of existing datasets, GAN-based generation, and data augmentation techniques to create a comprehensive training set.

### 1.1. Initial Datasets

The project used two publicly available handwritten signature datasets, sourced from [Handwritten Signature Datasets on Kaggle](https://www.kaggle.com/datasets/ishanikathuria/handwritten-signature-datasets):

- **BHSig260-Hindi**: This dataset contains genuine and forged signatures in the Hindi script.  
- **CEDAR**: A widely used dataset containing genuine and forged signatures in English.

### 1.2. Preprocessing

The raw images from these datasets were preprocessed to create a consistent format for model training. The preprocessing pipeline involved the following steps:

- **Image Resizing**: All signature images were resized to a uniform size of 128×128 pixels.  
- **Grayscale Conversion**: The images were converted to grayscale to focus on the structural features of the signatures rather than color information.  
- **Genuine and Forged Separation**: Genuine and forged signatures were separated based on the naming conventions and directory structures of the original datasets.

### 1.3. Forged Signature Augmentation

To significantly increase the diversity and size of the forged signature dataset, a multi-faceted approach was used, combining GAN generation with traditional manipulation techniques.

- **GAN Generation**: A trained CycleGAN was used to produce approximately 3,000 synthetic forged signatures from the genuine samples. The `generate_fake_signatures()` function was used to create files named `generated_fake_{:06d}.png`.  
- **Manipulation Techniques**: To further diversify the dataset, several manipulation techniques were implemented:  
  - Geometric Distortion: Rotation, scaling, and shearing.  
  - Stroke Modification: Morphological operations to alter stroke thickness.  
  - Noise and Blur: Addition of Gaussian blur and random noise.  
  - Intensity Changes: Adjustments to contrast and brightness.  
  - Elastic Deformation: Application of dense displacement fields to mimic natural variations.  
- **Original Forged Samples**: Around 1,500 original forged signatures from the initial datasets were also copied and resized to be included in the final set.

### 1.4. Final Dataset Assembly and Split

The final dataset was assembled by combining genuine signatures with the augmented set of forged signatures.  

- **Composition of Fake Dataset**:  
  - GAN-generated: ~3,000 samples  
  - Original forgeries: ~1,500 samples  
  - Manipulated forgeries: ~1,000 samples  

- **Train/Val/Test Split**: A stratified split was performed to ensure that each class ("real" and "fake") was represented proportionally across the sets:  
  - Training Set: 70%  
  - Validation Set: 15%  
  - Test Set: 15%  

- **Directory Structure**: The data was organized into a directory structure compatible with Keras' `ImageDataGenerator`:  

train/real, train/fake
val/real, val/fake
test/real, test/fake


---

## 2. Model Selection

Three different models were used for distinct tasks in this project: signature generation, signature detection, and signature classification.

### 2.1. Signature Generation: CycleGAN
A CycleGAN was chosen for generating synthetic forged signatures due to its effectiveness in unpaired image-to-image translation.

### 2.2. Signature Detection: YOLOv11s
For detecting the location of signatures within a document, the YOLOv11s (You Only Look Once) model was selected. YOLO is a state-of-the-art, real-time object detection system known for its speed and accuracy.

### 2.3. Signature Classification: Convolutional Neural Network (CNN)
Several CNN architectures, including a baseline model, InceptionV3, and EfficientNetV2, were trained and evaluated to classify the detected signatures as either "real" or "fake".

---

## 3. Training Process

Each of the selected models underwent a specific training process tailored to its task.

### 3.1. CycleGAN Architecture and Training

The CycleGAN was trained to translate genuine signatures into realistic forgeries.

- **Generator Architecture**: A residual encoder-decoder with 6 residual blocks. The output layer used a tanh activation function, and the model operated on single-channel (grayscale) images.  
- **Discriminator Architecture**: A PatchGAN discriminator was used, which outputs a linear value (no final activation), styled after the LSGAN approach.  
- **Loss Functions**:  
  - Adversarial Loss: Least Squares GAN (LSGAN) loss, which uses Mean Squared Error (MSE).  
  - Cycle-Consistency Loss: Mean Absolute Error (MAE) to ensure structural integrity.  
  - Identity Loss: Mean Absolute Error (MAE) to preserve color and composition.  

- **Training Configuration**:  
  - Image Size: 128×128  
  - Batch Size: 32  
  - Epochs: 50  
  - Optimizer: Adam with a learning rate of 2×10−4 and beta_1 of 0.5.  
  - Checkpointing: A custom `CycleGANCheckpointResume` callback was used to save model weights and a `training_state.json` file periodically, allowing for the resumption of long training sessions.  

### 3.2. Hyperparameter Tuning with Keras Tuner

To optimize the performance of the signature classification models, Keras Tuner was used to perform hyperparameter tuning using the Hyperband algorithm.

#### InceptionV3 Hyperparameter Tuning

def model_builder_efficientnet(hp):
inputs = keras.Input(shape=(128, 128, 1))
x = layers.Lambda(lambda x: tf.repeat(x, 3, axis=-1))(inputs)
x = base_model_efficientnet(x, training=False)
x = layers.GlobalAveragePooling2D()(x)

hp_dropout_1 = hp.Float('dropout_1', min_value=0.2, max_value=0.7, step=0.1)
x = layers.Dropout(hp_dropout_1)(x)

hp_dense_units_1 = hp.Int('dense_units_1', min_value=16, max_value=256, step=32)
x = layers.Dense(hp_dense_units_1, activation='relu')(x)

outputs = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs, outputs)

hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
optimizer = keras.optimizers.Adam(learning_rate=hp_learning_rate)

# Add AUC metric
auc_metric = tf.keras.metrics.AUC(curve="ROC", name="auc")  
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', auc_metric]
)
return model


**Best Hyperparameters for EfficientNetV2:**  
- Learning rate: 0.01  
- First dropout rate: 0.3  
- First dense units: 112  

### 3.3. YOLOv11s Fine-Tuning

- **Dataset**: The model was fine-tuned on the Tobacco800 Signatures dataset, available on Roboflow at [Tobacco800-Signatures](https://universe.roboflow.com/trainsignature/tobacco800-signatures).  
- **Hyperparameter Tuning**: The `tune` method from the ultralytics library was used to optimize the learning rate (`lr0`) and the final learning rate factor (`lrf`).  
- **Training**: The model was trained for 150 epochs with a batch size of 16. Early stopping with a patience of 10 epochs was used to prevent overfitting.  

---

## 4. Evaluation

The performance of the models was evaluated at different stages of the pipeline.

### 4.1. Model Comparison

Here is a detailed breakdown of the performance of each classification model on the test set:

- **Baseline CNN**  
  - Test AUC: 0.8315  
  - Accuracy: 75.17%  
  - Precision: 73.13%  
  - Recall: 77.00%  
  - F1-Score: 75.02%  

- **InceptionV3**  
  - Test AUC: 0.9049  
  - Accuracy: 83.18%  
  - Precision: 81.06%  
  - Recall: 85.14%  
  - F1-Score: 83.05%  

- **EfficientNetV2**  
  - Test AUC: 0.8965  
  - Accuracy: 82.11%  
  - Precision: 79.12%  
  - Recall: 85.66%  
  - F1-Score: 82.26%  

Based on these results, InceptionV3 and EfficientNetV2 significantly outperformed the baseline CNN. While InceptionV3 had a slightly higher Test AUC, both models demonstrated strong and comparable performance.

### 4.2. End-to-End Signature Verification on an Unseen Document

The entire pipeline was evaluated on an unseen document provided via a URL. This end-to-end evaluation involved the following steps:

1. **Signature Detection**: The fine-tuned YOLOv11s model was used to detect the location of signatures in the document.  
2. **Signature Cropping**: The detected signature regions were cropped from the document.  
3. **Real vs. Fake Classification**: Each cropped signature was then passed to the trained CNN classification model, which classified it as either "real" or "fake".  

The final output of the system is a classification of the detected signatures from the unseen document, demonstrating the successful integration of the different models.

### 4.3. Final Testing on Unseen Data from Hugging Face

For a final, unbiased test, two random images were selected from the [Mels22/SigDetectVerifyFlow](https://huggingface.co/datasets/Mels22/SigDetectVerifyFlow) dataset on Hugging Face.  

- In this dataset, the label 1 corresponds to a forged (fake) signature, and the label 0 corresponds to a genuine (real) signature.  

The two images chosen for this test were:  
- Forged (Fake) Signature - Label 1: [View Image](https://huggingface.co/datasets/Mels22/SigDetectVerifyFlow/viewer/default/train%3Fp%3D9%26image-viewer%3D9205BD0EC993E10A9F9F25DE84F0BF391A1C067E)  
- Genuine (Real) Signature - Label 0: [View Image](https://huggingface.co/datasets/Mels22/SigDetectVerifyFlow/viewer/default/train%3Fp%3D12%26image-viewer%3D8511AFB4ECE67A9B8896BF95365168EBEDB93AB5)  

The complete pipeline was run on these two images, successfully detecting the signature in each and correctly classifying them according to their ground truth labels, validating the model's effectiveness on completely unseen data from a different source.
