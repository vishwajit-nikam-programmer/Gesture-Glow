import os
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight

# Set paths
data_dir = r"C:\Users\nikam\Music\Final Code\New dataset"
model_dir = r"C:\Users\nikam\Music\Final Code\Gesture Models"
os.makedirs(model_dir, exist_ok=True)

# Constants
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 50
NUM_CLASSES = 4  # Four gesture classes (lights_on, lights_off, fan_on, fan_off)
SEED = 42
WARMUP_EPOCHS = 5
MIXED_IMAGES_PATH = os.path.join(data_dir, 'mixed')  # Path to mixed images folder

# Prepare Data
df = []
# Add gesture classes
gesture_classes = ['lights_on', 'lights_off', 'fan_on', 'fan_off']
for class_name in gesture_classes:
    class_path = os.path.join(data_dir, class_name)
    if os.path.isdir(class_path):
        for img in os.listdir(class_path):
            df.append({'filename': os.path.join(class_path, img), 'class': class_name})

# Add mixed images as a separate class
if os.path.exists(MIXED_IMAGES_PATH) and os.path.isdir(MIXED_IMAGES_PATH):
    for img in os.listdir(MIXED_IMAGES_PATH):
        df.append({'filename': os.path.join(MIXED_IMAGES_PATH, img), 'class': 'mixed'})

df = pd.DataFrame(df)

# Print class distribution including mixed images
print("Class distribution:")
print(df['class'].value_counts())

# Get class weights excluding mixed images for training
gesture_df = df[df['class'].isin(gesture_classes)]
print("\nGesture class distribution:")
print(gesture_df['class'].value_counts())

print("Sample DataFrame:\n", df.head())
print("Class distribution:\n", df['class'].value_counts())

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
fold_no = 1
accuracy_per_fold = []
loss_per_fold = []

for train_index, val_index in skf.split(df['filename'], df['class']):
    print(f"\n🔁 Training for Fold {fold_no}")

    train_df = df.iloc[train_index]
    val_df = df.iloc[val_index]

    # Calculate class weights only for gesture classes
    gesture_classes = ['lights_on', 'lights_off', 'fan_on', 'fan_off']
    train_gesture_df = train_df[train_df['class'].isin(gesture_classes)]
    
    class_weights = compute_class_weight('balanced',
                                        classes=np.unique(train_gesture_df['class']),
                                        y=train_gesture_df['class'])
    class_weights = dict(enumerate(class_weights))
    
    # Print class weights
    print("\nClass weights:", class_weights)

    # Enhanced Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        zoom_range=0.3,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        channel_shift_range=20,
        shear_range=0.2,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Filter out mixed images for training
    train_gesture_df = train_df[train_df['class'].isin(['lights_on', 'lights_off', 'fan_on', 'fan_off'])]
    
    train_generator = train_datagen.flow_from_dataframe(
        train_gesture_df,
        x_col='filename',
        y_col='class',
        target_size=IMG_SIZE,
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED
    )
    # Filter out mixed images for validation
    val_gesture_df = val_df[val_df['class'].isin(['lights_on', 'lights_off', 'fan_on', 'fan_off'])]
    
    val_generator = val_datagen.flow_from_dataframe(
        val_gesture_df,
        x_col='filename',
        y_col='class',
        target_size=IMG_SIZE,
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Build Model with improved architecture
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
    # Add custom head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Learning rate schedule with warmup
    def lr_schedule(epoch):
        if epoch < WARMUP_EPOCHS:
            return 1e-4 * (epoch + 1) / WARMUP_EPOCHS
        else:
            return 1e-4 * 0.95 ** (epoch - WARMUP_EPOCHS)

    # Compile with improved optimizer
    optimizer = Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=optimizer, 
                 loss='categorical_crossentropy', 
                 metrics=['accuracy', 'top_k_categorical_accuracy'])

    # Callbacks with improved settings
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, min_delta=0.001),
        ModelCheckpoint(f'{model_dir}/best_model_fold_{fold_no}.keras', 
                       save_best_only=True,
                       monitor='val_accuracy',
                       mode='max'),
        ReduceLROnPlateau(monitor='val_loss', 
                         factor=0.3, 
                         patience=4, 
                         min_lr=1e-7,
                         verbose=1),
        LearningRateScheduler(lr_schedule),
        TensorBoard(log_dir=f'{model_dir}/logs/fold_{fold_no}', 
                   histogram_freq=1,
                   write_graph=True,
                   write_images=True)
    ]

    # Train with class weights
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # Evaluate
    scores = model.evaluate(val_generator)
    print(f"✅ Fold {fold_no} — Loss: {scores[0]:.4f} — Accuracy: {scores[1]*100:.2f}%")
    accuracy_per_fold.append(scores[1])
    loss_per_fold.append(scores[0])

    # Optional: Classification report
    val_preds = model.predict(val_generator)
    y_pred = np.argmax(val_preds, axis=1)
    y_true = val_generator.classes
    print("Classification Report:\n", classification_report(y_true, y_pred, target_names=val_generator.class_indices.keys()))

    fold_no += 1

# Summary
print("\n📊 Cross-validation results:")
for i in range(5):
    print(f"Fold {i+1} — Loss: {loss_per_fold[i]:.4f} — Accuracy: {accuracy_per_fold[i]*100:.2f}%")

print(f"\n✅ Average Accuracy: {np.mean(accuracy_per_fold)*100:.2f}%")
print(f"❌ Average Loss: {np.mean(loss_per_fold):.4f}")