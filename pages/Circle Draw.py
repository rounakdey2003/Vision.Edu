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
        st.subheader("Circle Drawing")
    with col3:
        pass

with st.expander('Functions'):
    col1, col2 = st.columns(2)
    with col1:
        st.help(cv.circle)
    with col2:
        pass

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
            center_x = st.slider('Center X', min_value=0, max_value=opencv_image.shape[1], value=100, step=1)
            center_y = st.slider('Center Y', min_value=0, max_value=opencv_image.shape[0], value=150, step=1)
        with col2:
            radius = st.slider('Radius', min_value=1, max_value=200, value=50, step=1)
            thickness = st.slider('Thickness', min_value=1, max_value=20, value=2, step=1)
        
        circle_image = cv.circle(opencv_image.copy(), center=(center_x, center_y), radius=radius, color=(0, 255, 0), thickness=thickness)
        st.image(circle_image, channels="BGR")
        
        success, encoded_img = cv.imencode('.png', circle_image)
        if success:
            img_bytes = BytesIO(encoded_img.tobytes())
            st.download_button(
                label="Download Image",
                data=img_bytes.getvalue(),
                file_name=f"{uploaded_file.name.split('.')[0]}_circle.png",
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
        
# Draw circle with center=({center_x}, {center_y}), radius={radius}, thickness={thickness}
circle = cv2.circle(data.copy(), 
                   center=({center_x}, {center_y}), 
                   radius={radius}, 
                   color=(0, 255, 0),  # Green color
                   thickness={thickness})

# Display the image with circle
fig, ax = plt.subplots()
ax.imshow(circle)
ax.set_title('Circle: Center({center_x},{center_y}) R={radius} T={thickness}')
ax.axis('off')
plt.show()
        ''', language='python')
        
    elif file_extension in ['mp4', 'avi', 'mov']:
        st.subheader("Video Processing")
        
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
        
        col1, col2 = st.columns(2)
        with col1:
            center_x = st.slider('Center X', min_value=0, max_value=500, value=100, step=1)
            center_y = st.slider('Center Y', min_value=0, max_value=500, value=150, step=1)
        with col2:
            radius = st.slider('Radius', min_value=1, max_value=200, value=50, step=1)
            thickness = st.slider('Thickness', min_value=1, max_value=20, value=2, step=1)

        if uploaded_file is not None:
            data_video = cv.VideoCapture('temp_video.mp4')
            
            fps = int(data_video.get(cv.CAP_PROP_FPS))
            frame_count = int(data_video.get(cv.CAP_PROP_FRAME_COUNT))
            width = int(data_video.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(data_video.get(cv.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            output_filename = f"processed_{uploaded_file.name.split('.')[0]}_circle.mp4"
            out = cv.VideoWriter(output_filename, fourcc, fps, (width, height))
            
            frame_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            frame_num = 0
            
            while data_video.isOpened():
                ret, frame = data_video.read()
                
                if ret:
                    circle_frame = cv.circle(frame.copy(), center=(center_x, center_y), radius=radius, color=(0, 255, 0), thickness=thickness)
                    
                    out.write(circle_frame)
                    
                    frame_placeholder.image(circle_frame, channels="BGR", caption=f"Processing {frame_num + 1}/{frame_count} Frames at {fps} FPS")
                    
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
                        file_name=f"{uploaded_file.name.split('.')[0]}_circle.mp4",
                        mime="video/mp4"
                    )

                os.remove(output_filename)
        
        if os.path.exists("temp_video.mp4"):
            os.remove("temp_video.mp4")
        
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

        # Draw circle on frame
        circle_frame = cv2.circle(frame.copy(), 
                                 center=({center_x}, {center_y}), 
                                 radius={radius}, 
                                 color=(0, 255, 0),  # Green color
                                 thickness={thickness})

        # Display the frame with circle
        cv2.imshow('Circle Drawing', circle_frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(25) == ord('q'):
            break       
    else:
        break

# Release video capture object and close all windows
data_video.release()
out.release()
cv2.destroyAllWindows()
        ''', language='python')
        
    else:
        st.error("Unsupported file format. Please upload an image (jpg, jpeg, png) or video (mp4, avi, mov) file.")

else:
    st.info('Processing will be auto-detected based on the file.')