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
        st.subheader("Canny Edge Detection")
    with col3:
        pass

with st.expander('Functions'):
    col1, col2 = st.columns(2)
    with col1:
        st.help(cv.Canny)
    with col2:
        st.help(cv.cvtColor)  

st.divider()

uploaded_file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

st.divider()

if uploaded_file is not None:

    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension in ['jpg', 'jpeg', 'png']:
        st.subheader("Image Processing")
        
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv.imdecode(file_bytes, 1)
        
        low_threshold = st.slider('Low Threshold', min_value=0, max_value=255, value=100, step=1)
        high_threshold = st.slider('High Threshold', min_value=0, max_value=255, value=200, step=1)
        
        # Convert to grayscale first (Canny requires grayscale)
        gray = cv.cvtColor(opencv_image.copy(), cv.COLOR_BGR2GRAY)
        canny = cv.Canny(gray, low_threshold, high_threshold)
        st.image(canny)
        

        success, encoded_img = cv.imencode('.png', canny)
        if success:
            img_bytes = BytesIO(encoded_img.tobytes())
            st.download_button(
                label="Download Image",
                data=img_bytes.getvalue(),
                file_name=f"{uploaded_file.name.split('.')[0]}_canny.png",
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
        
# Convert to grayscale (Canny requires grayscale)
gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
canny = cv2.Canny(gray, {low_threshold}, {high_threshold})

# Display the edge detected image
fig, ax = plt.subplots()
ax.imshow(canny, cmap='gray')
ax.set_title('Canny Edge Detection - Thresholds: {low_threshold}/{high_threshold}')
ax.axis('off')
plt.show()
        ''', language='python')
        
    elif file_extension in ['mp4', 'avi', 'mov']:
        st.subheader("Video Processing")
        
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
        
        video_low_threshold = st.slider('Video Low Threshold', min_value=0, max_value=255, value=100, step=1)
        video_high_threshold = st.slider('Video High Threshold', min_value=0, max_value=255, value=200, step=1)

        if uploaded_file is not None:
            data_canny = cv.VideoCapture('temp_video.mp4')
            
            fps = int(data_canny.get(cv.CAP_PROP_FPS))
            frame_count = int(data_canny.get(cv.CAP_PROP_FRAME_COUNT))
            width = int(data_canny.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(data_canny.get(cv.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            output_filename = f"processed_{uploaded_file.name.split('.')[0]}_canny.mp4"
            out = cv.VideoWriter(output_filename, fourcc, fps, (width, height), isColor=False)
            
            frame_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            frame_num = 0
            
            while data_canny.isOpened():
                ret, frame = data_canny.read()
                
                if ret:
                    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    canny_frame = cv.Canny(grey, video_low_threshold, video_high_threshold)
                    
                    out.write(canny_frame)
                    
                    frame_placeholder.image(canny_frame, caption=f"Processing {frame_num + 1}/{frame_count} Frames at {fps} FPS")
                    
                    progress = (frame_num + 1) / frame_count
                    progress_bar.progress(progress)
                    
                    frame_num += 1
                else:
                    break
            
            data_canny.release()
            out.release()
            
            if os.path.exists(output_filename):
                with open(output_filename, "rb") as file:
                    st.download_button(
                        label="Download Video",
                        data=file.read(),
                        file_name=f"{uploaded_file.name.split('.')[0]}_canny.mp4",
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
data_canny = cv2.VideoCapture('video.mp4')

while data_canny.isOpened():
        
    # Read frame from video
    ret, frame = data_canny.read()
        
    # Check if frame was read successfully
    if ret:

        # Convert frame to grayscale
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        canny_frame = cv2.Canny(grey, {video_low_threshold}, {video_high_threshold})

        # Display the edge detected image
        cv2.imshow('Canny Edge Detection', canny_frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(25) == ord('q'):
            break       
    else:
        break

# Release video capture object and close all windows
data_canny.release()
out.release()
cv2.destroyAllWindows()
        ''', language='python')
        
    else:
        st.error("Unsupported file format. Please upload an image (jpg, jpeg, png) or video (mp4, avi, mov) file.")

else:
    st.info('Processing will be auto-detected based on the file.')