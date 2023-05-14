import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from helper import image_resize
from golf_model import get_movenet_model_poly
from golf_model import get_mediapipe_model_poly
from golf_model import get_accuracy

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles
pose = mp_pose.Pose()

DEMO_VIDEO = 'assets/demo.mp4'

st.title('Golfpose Analysis', anchor='center')
st.sidebar.title('Golfpose Analysis')
app_mode = 'Run on Video'
pose_framework = st.sidebar.selectbox('Choose Pose Estimation Framework',
                                      ['Mediapipe', 'Movenet']
                                      )

if app_mode == 'Run on Video':

    st.set_option('deprecation.showfileUploaderEncoding', False)

    st.sidebar.markdown('---')

    st.markdown(' ## Output')

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader(
        "Upload a video", type=["mp4", "mov", 'avi', 'asf', 'm4v', 'moov'])
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    if not video_file_buffer:

        vid = cv2.VideoCapture(DEMO_VIDEO)
        tfflie.name = DEMO_VIDEO

    else:
        tfflie.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tfflie.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    st.sidebar.text('Input Video')
    st.sidebar.video(tfflie.name)
    fps = 0
    i = 0
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
# ROW 1
    with kpi1:
        st.markdown("**FrameRate**")
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown("**Pose Estimation**")
        kpi2_text = st.markdown("Movenet")

    with kpi3:
        st.markdown("**Output Label**")
        kpi3_text = st.markdown("0")

    with kpi4:
        st.markdown("**Confidence Level**")
        kpi4_text = st.markdown("0")

    st.markdown("<hr/>", unsafe_allow_html=True)
    prevTime = 0
    # Initialize dictionary to store highest confidence level frame for each predicted label
    highest_conf_frames = {}

    # Initialize counter for number of frames with "Mid-Follow-Through" label
    follow_through_count = 0

    # Define the order in which to display labels
    label_order = [1, 2, 3, 4, 5]
    while vid.isOpened():

        i += 1
        ret, frame = vid.read()
        if not ret:
            # Loop through all the video
            # vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime
        # Draw the pose landmarks on the image
        annotated_image = frame.copy()
        # Dashboard
        if pose_framework == 'Movenet':
            pose_result = get_movenet_model_poly(frame)

        elif pose_framework == 'Mediapipe':
            pose_result = get_mediapipe_model_poly(frame)
            if (pose_result is None):
                continue

        kpi1_text.write(
            f"<h4 style='margin-left: 1rem; color: red;'>{int(fps)}</h4>", unsafe_allow_html=True)
        kpi2_text.write(
            f"<h4 style='margin-left: 1rem; color: red;'>{pose_framework}</h4>", unsafe_allow_html=True)
        kpi3_text.write(
            f"<h4 style='margin-left: 1rem; color: red;'>{pose_result[0]}</h4>", unsafe_allow_html=True)
        kpi4_text.write(
            f"<h4 style='margin-left: 1rem; color: red;'>{pose_result[1]:.2f}</h4>", unsafe_allow_html=True)
        # frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
        # frame = image_resize(image=frame, width=640)
        stframe.image(frame, channels='BGR', use_column_width=True)

        # Check if current frame has highest confidence level for predicted label
        if pose_result[2] not in highest_conf_frames or pose_result[1] > highest_conf_frames[pose_result[2]][0]:
            # Update dictionary with current frame as new highest confidence level frame
            highest_conf_frames[pose_result[2]] = (
                pose_result[1], annotated_image)

        # Increment counter for "Mid-Follow-Through" label frames
        if pose_result[2] == 5:
            follow_through_count += 1

            # Stop displaying "Mid-Follow-Through" label frames after 20 frames have been displayed

    # END LOOP
    # SUMMARY CONTENT
    st.markdown(' ## Summary:')
    st.markdown(f' ### Model Used: SVM Radial Basis Function')
    accuracy = get_accuracy(pose_framework)
    st.markdown(f' ### {pose_framework} Accuracy: {accuracy*100:.2f}%')
    st.markdown(' ### Highest Confidence Level:')
    for label in label_order:
        if label in highest_conf_frames:
            (confidence, frame) = highest_conf_frames[label]
            label_name = {
                1: "Address",
                2: "Impact",
                3: "Mid-Backswing",
                4: "Mid-Downswing",
                5: "Mid-Follow-Through"
            }[label]
            st.markdown(
                f"<h4 style='margin-left: 1rem;'>Label {label} ({label_name}) has similarity level of {confidence:.2f}%</h4>""", unsafe_allow_html=True)
            st.image(frame, use_column_width=True)

    vid.release()
    cv2.destroyAllWindows()
