import os
import ssl
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, recall_score

# =========================================================
# 0. CONFIGURATION & SETUP
# =========================================================
IMG_SIZE = 300
BATCH_SIZE = 32
EPOCHS = 70
SEED = 42
PREDICTION_THRESHOLD = 0.3

# Set Seeds for Reproducibility
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

# Fix SSL context for Mac
ssl._create_default_https_context = ssl._create_unverified_context

# =========================================================
# 1. DATA PREPARATION HELPERS
# =========================================================
def find_project_root(target="slide_images"):
    current = os.getcwd()
    while True:
        if target in os.listdir(current): return current
        parent = os.path.dirname(current)
        if parent == current: break 
        current = parent
    desktop = os.path.expanduser("~/Desktop/Slide Critiques")
    if os.path.exists(os.path.join(desktop, target)): return desktop
    raise FileNotFoundError(f"Cannot find '{target}' folder.")

def get_data_paths():
    base_dir = find_project_root()
    return {
        "train_img": os.path.join(base_dir, "slide_images", "training_set"),
        "val_img": os.path.join(base_dir, "slide_images", "validation_set"),
        "train_csv": os.path.join(base_dir, "slide_images", "training_set_tags.csv"),
        "val_csv": os.path.join(base_dir, "slide_images", "validation_set_tags.csv")
    }

def load_and_clean_df(csv_path):
    df = pd.read_csv(csv_path)
    df["tags"] = df["tags"].apply(lambda x: [t.strip() for t in str(x).split(",") if t.strip()])
    return df

# =========================================================
# 2. CUSTOM DATA GENERATOR (SlideSequence)
# =========================================================
class SlideSequence(tf.keras.utils.Sequence):
    def __init__(self, df, directory, x_col, y_col, batch_size, target_size, 
                 class_indices, class_weights_map, shuffle=True):
        self.df = df.copy()
        self.directory = directory
        self.x_col = x_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.target_size = target_size
        self.load_size = (400, 400) # Load larger to allow cropping
        self.class_indices = class_indices
        self.num_classes = len(class_indices)
        self.class_weights_map = class_weights_map
        self.shuffle = shuffle
        
        # Filter files
        self.df['exists'] = self.df[self.x_col].apply(lambda x: os.path.exists(os.path.join(directory, x)))
        self.df = self.df[self.df['exists']].reset_index(drop=True)

        self.y_matrix = self._preprocess_labels()
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _preprocess_labels(self):
        matrix = np.zeros((len(self.df), self.num_classes), dtype='float32')
        for i, tags in enumerate(self.df[self.y_col]):
            for tag in tags:
                if tag in self.class_indices:
                    matrix[i, self.class_indices[tag]] = 1.0
        return matrix

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        batch_indexes = self.indexes[start:end]
        
        batch_x = []
        batch_sample_weights = []
        batch_y = self.y_matrix[batch_indexes]
        
        for i, idx in enumerate(batch_indexes):
            filename = self.df.iloc[idx][self.x_col]
            img_path = os.path.join(self.directory, filename)
            
            # Load Image
            try:
                img = load_img(img_path, target_size=self.load_size)
                img = img_to_array(img) / 255.0
            except:
                img = np.zeros((self.load_size[0], self.load_size[1], 3), dtype='float32')

            # Multi-Scale Logic
            if np.random.rand() > 0.5:
                # Resize
                crop = tf.image.resize(img, self.target_size).numpy()
            else:
                # Random Crop
                h, w = img.shape[:2]
                target_h, target_w = self.target_size
                if h > target_h and w > target_w:
                    top = np.random.randint(0, h - target_h)
                    left = np.random.randint(0, w - target_w)
                    crop = img[top:top+target_h, left:left+target_w, :]
                else:
                    crop = tf.image.resize(img, self.target_size).numpy()

            batch_x.append(crop)
            
            # Weights
            active_indices = np.where(batch_y[i] == 1)[0]
            if len(active_indices) > 0:
                weight = max([self.class_weights_map[k] for k in active_indices])
            else:
                weight = 1.0
            batch_sample_weights.append(weight)
            
        return np.array(batch_x), batch_y, np.array(batch_sample_weights)

def sequence_to_tf_dataset(sequence):
    """Wraps a Sequence in a tf.data.Dataset for performance."""
    def generator_func():
        for i in range(len(sequence)):
            yield sequence[i]

    sample_x, sample_y, sample_w = sequence[0]
    output_signature = (
        tf.TensorSpec(shape=sample_x.shape, dtype=tf.float32),
        tf.TensorSpec(shape=sample_y.shape, dtype=tf.float32),
        tf.TensorSpec(shape=sample_w.shape, dtype=tf.float32)
    )
    dataset = tf.data.Dataset.from_generator(generator_func, output_signature=output_signature)
    return dataset.prefetch(tf.data.AUTOTUNE)

# =========================================================
# 3. MAIN EXECUTION FLOW
# =========================================================

# --- A. Load Data Metadata ---
paths = get_data_paths()
train_df = load_and_clean_df(paths["train_csv"])
validation_df = load_and_clean_df(paths["val_csv"])

# --- B. Determine Class Indices & Weights ---
# Use a temporary generator just to get class indices easily
temp_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
temp_generator = temp_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=paths["train_img"],
    x_col="filename",
    y_col="tags",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)
train_indices = temp_generator.class_indices
labels_map = {v: k for k, v in train_indices.items()}

manual_weights = {
    "Strategic_Text": 3, "Data_Chart": 3, "Graphics_Visuals": 3,
    "Framework_Structure": 2, "Process_Flow": 5, "Appendix_Reference": 1, "Title_Transition": 1
}
class_weights_map = {idx: manual_weights.get(name, 1.0) for name, idx in train_indices.items()}
print(f"Class Weights Map: {class_weights_map}")

# --- C. Create Datasets ---
train_sequence = SlideSequence(
    df=train_df,
    directory=paths["train_img"],
    x_col="filename",
    y_col="tags",
    batch_size=BATCH_SIZE,
    target_size=(IMG_SIZE, IMG_SIZE),
    class_indices=train_indices,
    class_weights_map=class_weights_map,
    shuffle=True
)
train_dataset = sequence_to_tf_dataset(train_sequence)

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
validation_dataset = validation_datagen.flow_from_dataframe(
    dataframe=validation_df,
    directory=paths["val_img"],
    x_col="filename",
    y_col="tags",
    class_mode="categorical",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    shuffle=False, # Important for evaluation
    seed=SEED
)

# =========================================================
# 4. MODEL DEFINITION
# =========================================================
base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

def grayscale_pipe(x):
    gray = tf.image.rgb_to_grayscale(x)
    return tf.image.grayscale_to_rgb(gray)

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal"),
  tf.keras.layers.RandomZoom(0.1),
  tf.keras.layers.RandomContrast(0.1),
  tf.keras.layers.Lambda(grayscale_pipe)
])

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalMaxPooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(len(labels_map), activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

# =========================================================
# 5. TRAINING
# =========================================================
class MacroRecall(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, name='val_macro_recall'):
        super().__init__()
        self.validation_data = validation_data
        self.name = name

    def on_epoch_end(self, epoch, logs=None):
        val_gen = self.validation_data
        val_gen.reset()
        y_pred_probs = self.model.predict(val_gen, verbose=0)
        y_pred = (y_pred_probs > PREDICTION_THRESHOLD).astype(int) 
        val_gen.reset()
        
        # Get true labels safely
        all_y = []
        steps = len(val_gen)
        for _ in range(steps):
            _, y = next(val_gen)
            all_y.append(y)
        y_true = np.vstack(all_y)
        
        macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        logs[self.name] = macro_recall
        print(f" — val_macro_recall: {macro_recall:.4f}")

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=EPOCHS/10, min_lr=0.00001),
    MacroRecall(validation_data=validation_dataset),
    tf.keras.callbacks.ModelCheckpoint(filepath="best_recall_model.keras", monitor='val_macro_recall', mode='max', save_best_only=True, verbose=1)
]

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    callbacks=callbacks
)
model.save("last_epoch_model.keras")

# =========================================================
# 6. EVALUATION & ANALYSIS
# =========================================================
print("\n" + "="*50 + "\nSTARTING FINAL EVALUATION\n" + "="*50)

# Reload best model
best_model = tf.keras.models.load_model(
    "best_recall_model.keras",
    custom_objects={"grayscale_pipe": grayscale_pipe},
    safe_mode=False
) 

# Analysis Functions
def plot_error_balance(y_true, y_pred, class_names):
    tp = np.sum((y_true == 1) & (y_pred == 1), axis=0)
    fn = np.sum((y_true == 1) & (y_pred == 0), axis=0)
    fp = np.sum((y_true == 0) & (y_pred == 1), axis=0)

    x = np.arange(len(class_names))
    width = 0.6
    plt.figure(figsize=(12, 6))
    plt.bar(x, tp, width, color='#22c55e', label='Correct (True Pos)')
    plt.bar(x, fn, width, bottom=tp, color='#eab308', label='Missed (False Neg)')
    plt.bar(x, fp, width, bottom=tp+fn, color='#ef4444', label='Hallucinated (False Pos)')
    plt.ylabel('Number of Slides')
    plt.title('Error Balance Sheet (Custom Thresholds)')
    plt.xticks(x, class_names, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_confusion_heatmap(y_true, y_pred, class_names):
    num_classes = len(class_names)
    matrix = np.zeros((num_classes, num_classes))
    for i in range(num_classes):      
        for j in range(num_classes):  
            if i == j: continue 
            mask = (y_true[:, i] == 1) & (y_pred[:, j] == 1) & (y_true[:, j] == 0)
            matrix[i, j] = np.sum(mask)
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, cmap="Reds", xticklabels=class_names, yticklabels=class_names, fmt='g')
    plt.title("Cross-Contamination Matrix")
    plt.tight_layout()
    plt.show()

# --- Run Predictions ---
validation_dataset.reset()
y_true_list, y_pred_list = [], []
num_batches = int(math.ceil(validation_dataset.samples / BATCH_SIZE))
print(f"Generating predictions for {validation_dataset.samples} validation images...")

for i in range(num_batches):
    batch = next(validation_dataset)
    if len(batch) == 3: x, y, _ = batch
    else: x, y = batch
    y_true_list.append(y)
    y_pred_list.append(best_model.predict_on_batch(x))

y_true = np.vstack(y_true_list)[:validation_dataset.samples]
y_pred_probs = np.vstack(y_pred_list)[:validation_dataset.samples]

# --- Apply Thresholds ---
THRESHOLD_MAP = {
    'Appendix_Reference':  0.3,
    'Data_Chart':          0.3, 
    'Framework_Structure': 0.5,  
    'Graphics_Visuals':    0.4,
    'Process_Flow':        0.3,  
    'Strategic_Text':      0.5,  
    'Title_Transition':    0.5
}
class_names = list(labels_map.values())
threshold_vector = np.array([THRESHOLD_MAP.get(name, 0.5) for name in class_names])
y_pred_dynamic = (y_pred_probs > threshold_vector).astype(int)

# --- Report & Plots ---
print("\nClassification Report (Custom Thresholds):")
print(classification_report(y_true, y_pred_dynamic, target_names=class_names, zero_division=0))

plot_error_balance(y_true, y_pred_dynamic, class_names)
plot_confusion_heatmap(y_true, y_pred_dynamic, class_names)