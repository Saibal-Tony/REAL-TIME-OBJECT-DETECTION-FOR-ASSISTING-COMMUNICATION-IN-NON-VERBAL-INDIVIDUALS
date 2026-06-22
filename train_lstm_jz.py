import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# ── Config ───────────────────────────────────────
GESTURES        = ['J', 'Z']
SEQUENCES       = 30
SEQUENCE_LENGTH = 30
DATA_PATH       = 'JZ_Data'
MODEL_PATH      = 'jz_lstm_model.keras'

# ── Load data ─────────────────────────────────────
print('Loading data...')
X, y = [], []

for label_idx, gesture in enumerate(GESTURES):
    for seq in range(SEQUENCES):
        sequence = []
        for frame in range(SEQUENCE_LENGTH):
            npy = np.load(os.path.join(
                DATA_PATH, gesture, str(seq), f'{frame}.npy'))
            sequence.append(npy)
        X.append(sequence)
        y.append(label_idx)

X = np.array(X)  # shape (60, 30, 63)
y = to_categorical(y, len(GESTURES))

print(f'✅ Data shape: {X.shape}')
print(f'   Labels    : {y.shape}')

# ── Split ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y.argmax(axis=1))

print(f'   Train: {X_train.shape[0]} | Test: {X_test.shape[0]}')

# ── Build LSTM model ─────────────────────────────
model = Sequential([
    LSTM(128, return_sequences=True,
         input_shape=(SEQUENCE_LENGTH, 63),
         activation='tanh'),
    BatchNormalization(),
    Dropout(0.3),

    LSTM(64, return_sequences=True, activation='tanh'),
    BatchNormalization(),
    Dropout(0.3),

    LSTM(32, return_sequences=False, activation='tanh'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(len(GESTURES), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ── Callbacks ─────────────────────────────────────
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=30,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=0.00001,
        verbose=1
    ),
    ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# ── Train ─────────────────────────────────────────
print('\n▶️  Training LSTM...')
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=8,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# ── Evaluate ──────────────────────────────────────
print('\n📊 Evaluation:')
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f'   Test Accuracy: {acc*100:.1f}%')
print(f'   Test Loss    : {loss:.4f}')

# ── Confusion matrix ─────────────────────────────
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=GESTURES, yticklabels=GESTURES)
plt.title(f'Confusion Matrix — Accuracy: {acc*100:.1f}%')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('jz_confusion_matrix.png')
plt.show()

# ── Training curves ───────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['accuracy'],     label='Train')
ax1.plot(history.history['val_accuracy'], label='Val')
ax1.set_title('Accuracy')
ax1.legend()

ax2.plot(history.history['loss'],     label='Train')
ax2.plot(history.history['val_loss'], label='Val')
ax2.set_title('Loss')
ax2.legend()

plt.savefig('jz_training_curves.png')
plt.show()

print(f'\n✅ Model saved: {MODEL_PATH}')
print('\nClassification Report:')
print(classification_report(
    y_test.argmax(axis=1),
    y_pred.argmax(axis=1),
    target_names=GESTURES))