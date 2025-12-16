import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt ### visualization
#import our model we built in emotion_detection_model.py
emotion_model = tf.keras.models.load_model('models/emotion_model_fine_tuned_v2.keras')
# Show the model architecture
emotion_model.summary()
IMG_SIZE = 96  # resize images to IMG_SIZExIMG_SIZE
BATCH_SIZE = 64
NUM_CLASSES = 6 #we drop disgust to simplify model, as there isnt enough data on disgust to get good results
T = np.load("models/temperature.npy")


emotion_labels = ["angry","fear","happy","sad","surprise","neutral"]
img = cv2.imread("pog.png")
print(img.shape)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (96,96))
img = img.astype(np.float32)
img = np.expand_dims(img, axis=0)
print(emotion_model.predict(img))
print(emotion_labels[np.argmax(emotion_model.predict(img))])
logits = emotion_model.predict(img, verbose=0)     # shape (1, C)
probs  = tf.nn.softmax(logits / T).numpy()[0]          # calibrated probs
emotion = emotion_labels[int(np.argmax(probs))]
print(emotion)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset\\test",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    class_names=emotion_labels,
    shuffle=False,
)
'''
#Collect our TRUE label-predictions

y_true = [] #ground truth emotion index
y_pred = [] #predicted emotion index

for images, labels in test_ds:
    preds = emotion_model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)
#Construct + Display our Confusion Matrix

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=emotion_labels
)

plt.figure(figsize=(8, 8))
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix (Test Set)")
plt.show()

test_loss, test_acc = emotion_model.evaluate(test_ds)
print(test_acc)
'''