import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_docs.vis import embed
import numpy as np
import cv2

# Import matplotlib libraries
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

# Some modules to display an animation using imageio.
import imageio
from IPython.display import HTML, display

import mediapipe as mp
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense
import pandas as pd
import numpy as np

import math

import requests

# Load the dataset
dfMP = pd.read_csv('GolfMediaPipeData.csv')
dfM = pd.read_csv('GolfMovenetData.csv')
dfOP = pd.read_csv('GolfOpenPoseData.csv')

X = dfMP[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']]
X = X.values
y = dfMP[['Label']]
y = y.values

X1 = dfM[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']]
X1 = X1.values
y1 = dfM[['Label']]
y1 = y1.values

X2 = dfOP[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']]
X2 = X2.values
y2 = dfOP[['Label']]
y2 = y2.values

# Mediapipe RBF
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# initialize a StandardScaler object
scaler = StandardScaler()

# fit the scaler to the data and transform it
X_scaled = scaler.fit_transform(X)

# initialize an SVM model with linear kernel
clf = SVC(kernel='rbf', gamma=0.1, C=1, probability=True)

# fit the model to the scaled data
clf.fit(X_scaled, y)

# make predictions on the scaled data
y_pred = clf.predict(X_scaled)

print(y_pred)
accuracy = accuracy_score(y, y_pred)
print("Mediapipe Accuracy:", accuracy)

# Movenet Linear
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2)
# initialize a StandardScaler object
scaler1 = StandardScaler()

# fit the scaler to the data and transform it
X_scaled1 = scaler1.fit_transform(X1)

# initialize an SVM model with linear kernel
clf1 = SVC(kernel='linear', C=1, probability=True)

# fit the model to the scaled data
clf1.fit(X_scaled1, y1)

# make predictions on the scaled data
y_pred1 = clf1.predict(X_scaled1)

print(y_pred1)
accuracy1 = accuracy_score(y1, y_pred1)
print("Movenet Accuracy:", accuracy1)

# OpenPose RBF
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2)
# initialize a StandardScaler object
scaler2 = StandardScaler()

# fit the scaler to the data and transform it
X_scaled2 = scaler2.fit_transform(X2)

# initialize an SVM model with linear kernel
clf2 = SVC(kernel='rbf', gamma=0.1, C=1, probability=True)

# fit the model to the scaled data
clf2.fit(X_scaled2, y2)

# make predictions on the scaled data
y_pred2 = clf2.predict(X_scaled2)

print(y_pred2)
accuracy2 = accuracy_score(y2, y_pred2)
print("OpenPose Accuracy:", accuracy2)


def get_accuracy(pose_framework):
    if pose_framework == 'Mediapipe':
        return accuracy
    elif pose_framework == 'Movenet':
        return accuracy1
    elif pose_framework == 'OpenPose':
        return accuracy2


# @title Helper functions for visualization
left_shoulder = []
right_shoulder = []
left_elbow = []
right_elbow = []
left_wrist = []
right_wrist = []
left_hip = []
right_hip = []
left_knee = []
right_knee = []
left_ankle = []
right_ankle = []


def calculate_angles_movenet(firstPoint, midPoint, lastPoint):
    # Same formula from the Android App for consistency
    result = math.degrees(math.atan2(lastPoint[1] - midPoint[1], lastPoint[0] - midPoint[0]) - math.atan2(
        firstPoint[1] - midPoint[1], firstPoint[0] - midPoint[0]))
    result = abs(result)  # Angle should never be negative
    #
    if result > 180:
        result = 360.0 - result  # Always get the acute representation of the angle
    return result


# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Maps bones to a matplotlib color name.
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}


def _keypoints_and_edges_for_display(keypoints_with_scores,
                                     height,
                                     width,
                                     keypoint_threshold=0.11):
    """Returns high confidence keypoints and edges for visualization.

    Args:
      keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
        the keypoint coordinates and scores returned from the MoveNet model.
      height: height of the image in pixels.
      width: width of the image in pixels.
      keypoint_threshold: minimum confidence score for a keypoint to be
        visualized.

    Returns:
      A (keypoints_xy, edges_xy, edge_colors) containing:
        * the coordinates of all keypoints of all detected entities;
        * the coordinates of all skeleton edges of all detected entities;
        * the colors in which the edges should be plotted.
    """
    keypoints_all = []
    keypoint_edges_all = []
    edge_colors = []
    num_instances, _, _, _ = keypoints_with_scores.shape
    for idx in range(num_instances):
        kpts_x = keypoints_with_scores[0, idx, :, 1]
        kpts_y = keypoints_with_scores[0, idx, :, 0]
        kpts_scores = keypoints_with_scores[0, idx, :, 2]
        kpts_absolute_xy = np.stack(
            [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
        kpts_above_thresh_absolute = kpts_absolute_xy[
            kpts_scores > keypoint_threshold, :]
        keypoints_all.append(kpts_above_thresh_absolute)

        for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
            if (kpts_scores[edge_pair[0]] > keypoint_threshold and
                    kpts_scores[edge_pair[1]] > keypoint_threshold):
                x_start = kpts_absolute_xy[edge_pair[0], 0]
                y_start = kpts_absolute_xy[edge_pair[0], 1]
                x_end = kpts_absolute_xy[edge_pair[1], 0]
                y_end = kpts_absolute_xy[edge_pair[1], 1]
                line_seg = np.array([[x_start, y_start], [x_end, y_end]])
                keypoint_edges_all.append(line_seg)
                edge_colors.append(color)
    if keypoints_all:
        keypoints_xy = np.concatenate(keypoints_all, axis=0)
    else:
        keypoints_xy = np.zeros((0, 17, 2))

    if keypoint_edges_all:
        edges_xy = np.stack(keypoint_edges_all, axis=0)
    else:
        edges_xy = np.zeros((0, 2, 2))
    return keypoints_xy, edges_xy, edge_colors


def draw_prediction_on_image(
        image, keypoints_with_scores, crop_region=None, close_figure=False,
        output_image_height=None):
    """Draws the keypoint predictions on image.

    Args:
      image: A numpy array with shape [height, width, channel] representing the
        pixel values of the input image.
      keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
        the keypoint coordinates and scores returned from the MoveNet model.
      crop_region: A dictionary that defines the coordinates of the bounding box
        of the crop region in normalized coordinates (see the init_crop_region
        function below for more detail). If provided, this function will also
        draw the bounding box on the image.
      output_image_height: An integer indicating the height of the output image.
        Note that the image aspect ratio will be the same as the input image.

    Returns:
      A numpy array with shape [out_height, out_width, channel] representing the
      image overlaid with keypoint predictions.
    """
    height, width, channel = image.shape
    aspect_ratio = float(width) / height
    fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
    # To remove the huge white borders
    fig.tight_layout(pad=0)
    ax.margins(0)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.axis('off')

    im = ax.imshow(image)
    line_segments = LineCollection([], linewidths=(4), linestyle='solid')
    ax.add_collection(line_segments)
    # Turn off tick labels
    scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

    (keypoint_locs, keypoint_edges,
     edge_colors) = _keypoints_and_edges_for_display(
         keypoints_with_scores, height, width)

    line_segments.set_segments(keypoint_edges)
    line_segments.set_color(edge_colors)
    if keypoint_edges.shape[0]:
        line_segments.set_segments(keypoint_edges)
        line_segments.set_color(edge_colors)
    if keypoint_locs.shape[0]:
        scat.set_offsets(keypoint_locs)

    if crop_region is not None:
        xmin = max(crop_region['x_min'] * width, 0.0)
        ymin = max(crop_region['y_min'] * height, 0.0)
        rec_width = min(crop_region['x_max'], 0.99) * width - xmin
        rec_height = min(crop_region['y_max'], 0.99) * height - ymin
        rect = patches.Rectangle(
            (xmin, ymin), rec_width, rec_height,
            linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(
        fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    if output_image_height is not None:
        output_image_width = int(output_image_height / height * width)
        image_from_plot = cv2.resize(
            image_from_plot, dsize=(output_image_width, output_image_height),
            interpolation=cv2.INTER_CUBIC)
    return image_from_plot


def to_gif(images, fps):
    """Converts image sequence (4D numpy array) to gif."""
    imageio.mimsave('./animation.gif', images, fps=fps)
    return embed.embed_file('./animation.gif')


def progress(value, max=100):
    return HTML("""
      <progress
          value='{value}'
          max='{max}',
          style='width: 100%'
      >
          {value}
      </progress>
  """.format(value=value, max=max))


model_name = "movenet_lightning"

if "tflite" in model_name:
    if "movenet_lightning_f16" in model_name:
        url = "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite"
        response = requests.get(url)
        with open("model.tflite", "wb") as f:
            f.write(response.content)
        input_size = 192
    elif "movenet_thunder_f16" in model_name:
        url = "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format = tflite"
        response = requests.get(url)
        with open("model.tflite", "wb") as f:
            f.write(response.content)
        input_size = 256
    elif "movenet_lightning_int8" in model_name:
        url = "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format = tflite"
        response = requests.get(url)
        with open("model.tflite", "wb") as f:
            f.write(response.content)
        input_size = 192
    elif "movenet_thunder_int8" in model_name:
        url = "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/int8/4?lite-format = tflite"
        response = requests.get(url)
        with open("model.tflite", "wb") as f:
            f.write(response.content)
        input_size = 256
    else:
        raise ValueError("Unsupported model name: %s" % model_name)

    # Initialize the TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()

    def movenet(input_image):
        """Runs detection on an input image.

        Args:
          input_image: A [1, height, width, 3] tensor represents the input image
            pixels. Note that the height/width should already be resized and match the
            expected input resolution of the model before passing into this function.

        Returns:
          A [1, 1, 17, 3] float numpy array representing the predicted keypoint
          coordinates and scores.
        """
        # TF Lite format expects tensor type of uint8.
        input_image = tf.cast(input_image, dtype=tf.uint8)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
        # Invoke inference.
        interpreter.invoke()
        # Get the model prediction.
        keypoints_with_scores = interpreter.get_tensor(
            output_details[0]['index'])
        return keypoints_with_scores

else:
    if "movenet_lightning" in model_name:
        module = hub.load(
            "https://tfhub.dev/google/movenet/singlepose/lightning/4")
        input_size = 192
    elif "movenet_thunder" in model_name:
        module = hub.load(
            "https://tfhub.dev/google/movenet/singlepose/thunder/4")
        input_size = 256
    else:
        raise ValueError("Unsupported model name: %s" % model_name)

    def movenet(input_image):
        """Runs detection on an input image.

        Args:
          input_image: A [1, height, width, 3] tensor represents the input image
            pixels. Note that the height/width should already be resized and match the
            expected input resolution of the model before passing into this function.

        Returns:
          A [1, 1, 17, 3] float numpy array representing the predicted keypoint
          coordinates and scores.
        """
        model = module.signatures['serving_default']

        # SavedModel format expects tensor type of int32.
        input_image = tf.cast(input_image, dtype=tf.int32)
        # Run model inference.
        outputs = model(input_image)
        # Output is a [1, 1, 17, 3] tensor.
        keypoints_with_scores = outputs['output_0'].numpy()
        return keypoints_with_scores

    def movenet(input_image, keypoint_indices):
        """Runs detection on an input image and prints the x and y coordinates of
        the keypoints at index keypoint_indices.

        Args:
          input_image: A [1, height, width, 3] tensor represents the input image
            pixels. Note that the height/width should already be resized and match the
            expected input resolution of the model before passing into this function.
          keypoint_indices: A list of integers representing the indices of the keypoints
            to print the x and y coordinates of.

        Returns:
          A [1, 1, 17, 3] float numpy array representing the predicted keypoint
          coordinates and scores.
        """
        model = module.signatures['serving_default']

        # SavedModel format expects tensor type of int32.
        input_image = tf.cast(input_image, dtype=tf.int32)
        # Run model inference.
        outputs = model(input_image)
        # Output is a [1, 1, 17, 3] tensor.
        keypoints_with_scores = outputs['output_0'].numpy()

        for index in keypoint_indices:
            x, y = keypoints_with_scores[0, 0, index, :2]
            if index == 5:
                left_shoulder = [x, y]
            if index == 6:
                right_shoulder = [x, y]
            if index == 7:
                left_elbow = [x, y]
            if index == 8:
                right_elbow = [x, y]
            if index == 9:
                left_wrist = [x, y]
            if index == 10:
                right_wrist = [x, y]
            if index == 11:
                left_hip = [x, y]
            if index == 12:
                right_hip = [x, y]
            if index == 13:
                left_knee = [x, y]
            if index == 14:
                right_knee = [x, y]
            if index == 15:
                left_ankle = [x, y]
            if index == 16:
                right_ankle = [x, y]

            # print(f"Keypoint {index} - x:{x}, y:{y}")

        left_elbow_angles = calculate_angles_movenet(
            left_wrist, left_elbow, left_shoulder)
        right_elbow_angles = calculate_angles_movenet(
            right_wrist, right_elbow, right_shoulder)
        left_shoulder_angles = calculate_angles_movenet(
            left_elbow, left_shoulder, left_hip)
        right_shoulder_angles = calculate_angles_movenet(
            right_elbow, right_shoulder, right_hip)
        left_hip_angles = calculate_angles_movenet(
            left_shoulder, left_hip, left_knee)
        right_hip_angles = calculate_angles_movenet(
            right_shoulder, right_hip, right_knee)
        left_knee_angles = calculate_angles_movenet(
            left_hip, left_knee, left_ankle)
        right_knee_angles = calculate_angles_movenet(
            right_hip, right_knee, right_ankle)

        # if (left_elbow_angles is None or right_elbow_angles is None or left_shoulder_angles is None or
        #     right_shoulder_angles is None or left_hip_angles is None or right_hip_angles is None or
        #     left_knee_angles is None or right_knee_angles is None):
        #     print("Skipped - some angles are null.")
        # else:
        #   golfdataset = pd.DataFrame([[left_elbow_angles, right_elbow_angles, left_shoulder_angles, right_shoulder_angles, left_hip_angles, right_hip_angles, left_knee_angles, right_knee_angles, 5]], columns=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', "Label"])
        #   golfdataset.to_csv('GolfMovenetData.csv', mode='a', index=False, header=False)

        return keypoints_with_scores

# left_elbow_angles = calculate
# right_elbow_angles = []
# left_shoulder_angles = []
# right_shoulder_angles = []
# left_hip_angles = []
# right_hip_angles = []
# left_knee_angles = []
# right_knee_angles = []

# L and R Angle:
# *Elbow = Wrist, Elbow, Shoulder
# *Shoulder = Elbow, Shoulder, Hip
# *Hips = Shoulder, Hip, Knee
# *Knee = Hip, Knee, Heel


def extract_angles(input_image, keypoint_indices):
    """Runs detection on an input image and prints the x and y coordinates of
    the keypoints at index keypoint_indices.

    Args:
      input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.
      keypoint_indices: A list of integers representing the indices of the keypoints
        to print the x and y coordinates of.

    Returns:
      A [1, 1, 17, 3] float numpy array representing the predicted keypoint
      coordinates and scores.
    """
    model = module.signatures['serving_default']

    # SavedModel format expects tensor type of int32.
    input_image = tf.cast(input_image, dtype=tf.int32)
    # Run model inference.
    outputs = model(input_image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoints_with_scores = outputs['output_0'].numpy()

    for index in keypoint_indices:
        x, y = keypoints_with_scores[0, 0, index, :2]
        if index == 5:
            left_shoulder = [x, y]
        if index == 6:
            right_shoulder = [x, y]
        if index == 7:
            left_elbow = [x, y]
        if index == 8:
            right_elbow = [x, y]
        if index == 9:
            left_wrist = [x, y]
        if index == 10:
            right_wrist = [x, y]
        if index == 11:
            left_hip = [x, y]
        if index == 12:
            right_hip = [x, y]
        if index == 13:
            left_knee = [x, y]
        if index == 14:
            right_knee = [x, y]
        if index == 15:
            left_ankle = [x, y]
        if index == 16:
            right_ankle = [x, y]

        # print(f"Keypoint {index} - x:{x}, y:{y}")

    left_elbow_angles = calculate_angles_movenet(
        left_wrist, left_elbow, left_shoulder)
    right_elbow_angles = calculate_angles_movenet(
        right_wrist, right_elbow, right_shoulder)
    left_shoulder_angles = calculate_angles_movenet(
        left_elbow, left_shoulder, left_hip)
    right_shoulder_angles = calculate_angles_movenet(
        right_elbow, right_shoulder, right_hip)
    left_hip_angles = calculate_angles_movenet(
        left_shoulder, left_hip, left_knee)
    right_hip_angles = calculate_angles_movenet(
        right_shoulder, right_hip, right_knee)
    left_knee_angles = calculate_angles_movenet(
        left_hip, left_knee, left_ankle)
    right_knee_angles = calculate_angles_movenet(
        right_hip, right_knee, right_ankle)

    if (left_elbow_angles is None or right_elbow_angles is None or left_shoulder_angles is None or
        right_shoulder_angles is None or left_hip_angles is None or right_hip_angles is None or
            left_knee_angles is None or right_knee_angles is None):
        print("Skipped - some angles are null.")

    return [[left_elbow_angles, right_elbow_angles, left_shoulder_angles, right_shoulder_angles, left_hip_angles, right_hip_angles, left_knee_angles, right_knee_angles]]


def get_movenet_model_poly(frame):
    input_image = cv2.resize(frame, (input_size, input_size))
    input_image = tf.expand_dims(input_image, axis=0)

    # Run the model on the input image to get the keypoints and scores
    keypoint_indices = [0, 1, 2, 3, 4, 5, 6,
                        7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    keypoints_with_scores = movenet(input_image, keypoint_indices)
    extract_angle = extract_angles(input_image, keypoint_indices)

    test_scaled_value = scaler.transform(extract_angle)
    test_probabilities = clf1.predict_proba(test_scaled_value)
    test_prediction = np.argmax(test_probabilities)
    test_confidence = test_probabilities[0][test_prediction] * 100
    test_prediction += 1  # add 1 to test_prediction
    label_name = {
        1: "Address",
        2: "Impact",
        3: "Mid-Backswing",
        4: "Mid-Downswing",
        5: "Mid-Follow-Through"
    }[test_prediction]

    return label_name, test_confidence, test_prediction


mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles
pose = mp_pose.Pose()


def calculate_angles(firstPoint, midPoint, lastPoint):
    # Same formula from the Android App for consistency
    result = math.degrees(math.atan2(lastPoint.y - midPoint.y, lastPoint.x - midPoint.x) -
                          math.atan2(firstPoint.y - midPoint.y, firstPoint.x - midPoint.x))
    result = abs(result)  # Angle should never be negative
    #
    if result > 180:
        result = 360.0 - result  # Always get the acute representation of the angle
    return result


def extract_angles_test(results):
    if results.pose_landmarks is None:
        return None

    # Calculates the essential angles for each image and adds them to the array
    left_elbow_angles = calculate_angles(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST], results.pose_landmarks.landmark[
                                         mp_pose.PoseLandmark.LEFT_ELBOW], results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER])
    right_elbow_angles = calculate_angles(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST], results.pose_landmarks.landmark[
                                          mp_pose.PoseLandmark.RIGHT_ELBOW], results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER])
    left_shoulder_angles = calculate_angles(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW], results.pose_landmarks.landmark[
                                            mp_pose.PoseLandmark.LEFT_SHOULDER], results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP])
    right_shoulder_angles = calculate_angles(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW], results.pose_landmarks.landmark[
                                             mp_pose.PoseLandmark.RIGHT_SHOULDER], results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP])
    left_hip_angles = calculate_angles(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER], results.pose_landmarks.landmark[
                                       mp_pose.PoseLandmark.LEFT_HIP], results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE])
    right_hip_angles = calculate_angles(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER], results.pose_landmarks.landmark[
                                        mp_pose.PoseLandmark.RIGHT_HIP], results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE])
    left_knee_angles = calculate_angles(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP], results.pose_landmarks.landmark[
                                        mp_pose.PoseLandmark.LEFT_KNEE], results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL])
    right_knee_angles = calculate_angles(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP], results.pose_landmarks.landmark[
                                         mp_pose.PoseLandmark.RIGHT_KNEE], results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL])

    return [[left_elbow_angles, right_elbow_angles, left_shoulder_angles, right_shoulder_angles, left_hip_angles, right_hip_angles, left_knee_angles, right_knee_angles]]


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Set minimum confidence levels
min_detection_confidence = 0.5
min_tracking_confidence = 0.5


def get_mediapipe_model_poly(frame):
    with mp_pose.Pose(min_detection_confidence=min_detection_confidence,
                      min_tracking_confidence=min_tracking_confidence) as pose:
        results = pose.process(frame)
        test_value = extract_angles_test(results)
        print(test_value)
        test_scaled_value = scaler.transform(test_value)
        # print(test_scaled_value)
        test_probabilities = clf.predict_proba(test_scaled_value)
        test_prediction = np.argmax(test_probabilities)
        test_confidence = test_probabilities[0][test_prediction] * 100
        test_prediction += 1  # add 1 to test_prediction
        label_name = {
            1: "Address",
            2: "Impact",
            3: "Mid-Backswing",
            4: "Mid-Downswing",
            5: "Mid-Follow-Through"
        }[test_prediction]

    if not results.pose_landmarks or len(results.pose_landmarks.landmark) != 33:
        return None
    return label_name, test_confidence, test_prediction
