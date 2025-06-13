import streamlit as st
import cv2 as cv
import numpy as np
import os
from io import BytesIO

st.set_page_config(
    page_title="Image/Video Processing",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="expanded",
)

with st.container(height=100):
    col1, col2, col3 = st.columns(3)
    with col1:
        pass
    with col2:
        st.subheader("Contour Detection")
    with col3:
        pass

with st.expander('Functions'):
    col1, col2 = st.columns(2)
    with col1:
        st.help(cv.findContours)
    with col2:
        st.help(cv.drawContours)  

st.divider()

uploaded_file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

st.divider()

if uploaded_file is not None:

    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension in ['jpg', 'jpeg', 'png']:
        st.subheader("Image Processing")
        
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv.imdecode(file_bytes, 1)
        
        col1, col2 = st.columns(2)
        with col1:
            contour_mode = st.selectbox('Contour Mode', 
                                       options=[cv.RETR_EXTERNAL, cv.RETR_LIST, cv.RETR_CCOMP, cv.RETR_TREE],
                                       format_func=lambda x: {cv.RETR_EXTERNAL: 'External', cv.RETR_LIST: 'List', 
                                                              cv.RETR_CCOMP: 'CComp', cv.RETR_TREE: 'Tree'}[x],
                                       index=3)
            thickness = st.slider('Thickness', min_value=1, max_value=10, value=2, step=1)
        with col2:
            contour_method = st.selectbox('Contour Method',
                                         options=[cv.CHAIN_APPROX_NONE, cv.CHAIN_APPROX_SIMPLE],
                                         format_func=lambda x: {cv.CHAIN_APPROX_NONE: 'None', cv.CHAIN_APPROX_SIMPLE: 'Simple'}[x],
                                         index=1)
            contour_idx = st.slider('Contour Index (-1 for all)', min_value=-1, max_value=100, value=-1, step=1)
        
        mode_name = {cv.RETR_EXTERNAL: 'cv2.RETR_EXTERNAL', cv.RETR_LIST: 'cv2.RETR_LIST', 
                     cv.RETR_CCOMP: 'cv2.RETR_CCOMP', cv.RETR_TREE: 'cv2.RETR_TREE'}[contour_mode]
        method_name = {cv.CHAIN_APPROX_NONE: 'cv2.CHAIN_APPROX_NONE', cv.CHAIN_APPROX_SIMPLE: 'cv2.CHAIN_APPROX_SIMPLE'}[contour_method]
        
        grey = cv.cvtColor(opencv_image.copy(), cv.COLOR_BGR2GRAY)
        contours, hierarchy = cv.findContours(grey, mode=contour_mode, method=contour_method)
        max_contour_idx = len(contours) - 1 if contours else 0
        valid_contour_idx = contour_idx if contour_idx == -1 or (0 <= contour_idx <= max_contour_idx) else -1
        contour = cv.drawContours(opencv_image.copy(), contours, contourIdx=valid_contour_idx, color=(0,255,0), thickness=thickness)
        st.image(contour, channels="BGR")
        

        success, encoded_img = cv.imencode('.png', contour)
        if success:
            img_bytes = BytesIO(encoded_img.tobytes())
            st.download_button(
                label="Download Image",
                data=img_bytes.getvalue(),
                file_name=f"{uploaded_file.name.split('.')[0]}_contours.png",
                mime="image/png"
            )
        
        st.divider()

        st.subheader("Code")

        st.code(f'''
# Import necessary libraries
import cv2
import matplotlib.pyplot as plt

# Load image file
data = cv2.imread('image.jpg')
        
# Convert to grayscale
grey = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

# Find contours
contours, hierarchy = cv2.findContours(grey, 
                                      mode={mode_name}, 
                                      method={method_name})

# Draw contours
contour = cv2.drawContours(data.copy(), 
                          contours, 
                          contourIdx={valid_contour_idx}, 
                          color=(0, 255, 0),  # Green color
                          thickness={thickness})

# Display the contour image
fig, ax = plt.subplots()
ax.imshow(contour)
ax.set_title('Contours: Mode={mode_name.split(".")[-1]}, Method={method_name.split(".")[-1]}, T={thickness}')
ax.axis('off')
plt.show()
        ''', language='python')
        
    elif file_extension in ['mp4', 'avi', 'mov']:
        st.subheader("Video Processing")
        
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
        
        col1, col2 = st.columns(2)
        with col1:
            video_contour_mode = st.selectbox('Video Contour Mode', 
                                           options=[cv.RETR_EXTERNAL, cv.RETR_LIST, cv.RETR_CCOMP, cv.RETR_TREE],
                                           format_func=lambda x: {cv.RETR_EXTERNAL: 'External', cv.RETR_LIST: 'List', 
                                                                  cv.RETR_CCOMP: 'CComp', cv.RETR_TREE: 'Tree'}[x],
                                           index=3)
            video_thickness = st.slider('Video Thickness', min_value=1, max_value=10, value=2, step=1)
        with col2:
            video_contour_method = st.selectbox('Video Contour Method',
                                             options=[cv.CHAIN_APPROX_NONE, cv.CHAIN_APPROX_SIMPLE],
                                             format_func=lambda x: {cv.CHAIN_APPROX_NONE: 'None', cv.CHAIN_APPROX_SIMPLE: 'Simple'}[x],
                                             index=1)
            video_contour_idx = st.slider('Video Contour Index (-1 for all)', min_value=-1, max_value=100, value=-1, step=1)

        if uploaded_file is not None:
            data_video = cv.VideoCapture('temp_video.mp4')
            
            fps = int(data_video.get(cv.CAP_PROP_FPS))
            frame_count = int(data_video.get(cv.CAP_PROP_FRAME_COUNT))
            width = int(data_video.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(data_video.get(cv.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            output_filename = f"processed_{uploaded_file.name.split('.')[0]}_contours.mp4"
            out = cv.VideoWriter(output_filename, fourcc, fps, (width, height))
            
            frame_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            frame_num = 0
            
            while data_video.isOpened():
                ret, frame = data_video.read()
                
                if ret:
                    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    contours, hierarchy = cv.findContours(grey, mode=video_contour_mode, method=video_contour_method)
                    max_contour_idx = len(contours) - 1 if contours else 0
                    valid_contour_idx = video_contour_idx if video_contour_idx == -1 or (0 <= video_contour_idx <= max_contour_idx) else -1
                    contour_frame = cv.drawContours(frame.copy(), contours, contourIdx=valid_contour_idx, color=(0,255,0), thickness=video_thickness)
                    
                    out.write(contour_frame)
                    
                    frame_placeholder.image(contour_frame, channels="BGR", caption=f"Processing {frame_num + 1}/{frame_count} Frames at {fps} FPS")
                    
                    progress = (frame_num + 1) / frame_count
                    progress_bar.progress(progress)
                    
                    frame_num += 1
                else:
                    break
            
            data_video.release()
            out.release()
            
            if os.path.exists(output_filename):
                with open(output_filename, "rb") as file:
                    st.download_button(
                        label="Download Video",
                        data=file.read(),
                        file_name=f"{uploaded_file.name.split('.')[0]}_contours.mp4",
                        mime="video/mp4"
                    )

                os.remove(output_filename)
        
        if os.path.exists("temp_video.mp4"):
            os.remove("temp_video.mp4")
        
        video_mode_name = {cv.RETR_EXTERNAL: 'cv2.RETR_EXTERNAL', cv.RETR_LIST: 'cv2.RETR_LIST', 
                          cv.RETR_CCOMP: 'cv2.RETR_CCOMP', cv.RETR_TREE: 'cv2.RETR_TREE'}[video_contour_mode]
        video_method_name = {cv.CHAIN_APPROX_NONE: 'cv2.CHAIN_APPROX_NONE', cv.CHAIN_APPROX_SIMPLE: 'cv2.CHAIN_APPROX_SIMPLE'}[video_contour_method]
        
        st.divider()

        st.subheader("Code")

        st.code(f'''
# Import necessary libraries
import cv2
        
# Load video file
data_video = cv2.VideoCapture('video.mp4')

while data_video.isOpened():
        
    # Read frame from video
    ret, frame = data_video.read()
        
    # Check if frame was read successfully
    if ret:

        # Convert frame to grayscale
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find contours
        contours, hierarchy = cv2.findContours(grey, 
                                              mode={video_mode_name}, 
                                              method={video_method_name})

        # Draw contours
        contour_frame = cv2.drawContours(frame.copy(), 
                                        contours, 
                                        contourIdx={video_contour_idx}, 
                                        color=(0, 255, 0),  # Green color
                                        thickness={video_thickness})

        # Display the contour frame
        cv2.imshow('Contour Detection', contour_frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(25) == ord('q'):
            break       
    else:
        break

# Release video capture object and close all windows
data_video.release()
cv2.destroyAllWindows()
        ''', language='python')
        
    else:
        st.error("Unsupported file format. Please upload an image (jpg, jpeg, png) or video (mp4, avi, mov) file.")

else:
    st.info('Processing will be auto-detected based on the file.')