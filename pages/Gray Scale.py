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
        st.subheader("Gray Scale")
    with col3:
        pass

with st.expander('Functions'):
    col1, col2 = st.columns(2)
    with col1:
        st.help(cv.cvtColor)
    with col2:
        st.help(cv.COLOR_BGR2GRAY)  

st.divider()

uploaded_file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

st.divider()

if uploaded_file is not None:

    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension in ['jpg', 'jpeg', 'png']:
        st.subheader("Image Processing")
        
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv.imdecode(file_bytes, 1)
        
        brightness = st.slider('Brightness', min_value=-100, max_value=100, value=0, step=1)
        contrast = st.slider('Contrast', min_value=0.5, max_value=3.0, value=1.0, step=0.1)
        
        gray = cv.cvtColor(opencv_image.copy(), cv.COLOR_BGR2GRAY)
        
        adjusted_gray = cv.convertScaleAbs(gray, alpha=contrast, beta=brightness)
        st.image(adjusted_gray)
        

        success, encoded_img = cv.imencode('.png', adjusted_gray)
        if success:
            img_bytes = BytesIO(encoded_img.tobytes())
            st.download_button(
                label="Download Image",
                data=img_bytes.getvalue(),
                file_name=f"{uploaded_file.name.split('.')[0]}_gray.png",
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
gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

# Convert grayscale image to 3 channels
grey_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# Apply brightness and contrast adjustments
adjusted_gray = cv2.convertScaleAbs(grey_3ch, alpha={contrast}, beta={brightness})

# Display the adjusted grayscale image
fig, ax = plt.subplots()
ax.imshow(adjusted_gray)
ax.set_title('Gray Scale - Brightness: {brightness}, Contrast: {contrast}')
ax.axis('off')
plt.show()
        ''', language='python')
        
    elif file_extension in ['mp4', 'avi', 'mov']:
        st.subheader("Video Processing")
        
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
        
        video_brightness = st.slider('Video Brightness', min_value=-100, max_value=100, value=0, step=1)
        video_contrast = st.slider('Video Contrast', min_value=0.5, max_value=3.0, value=1.0, step=0.1)

        if uploaded_file is not None:
            data_grey = cv.VideoCapture('temp_video.mp4')
            
            fps = int(data_grey.get(cv.CAP_PROP_FPS))
            frame_count = int(data_grey.get(cv.CAP_PROP_FRAME_COUNT))
            width = int(data_grey.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(data_grey.get(cv.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            output_filename = f"processed_{uploaded_file.name.split('.')[0]}_gray.mp4"
            out = cv.VideoWriter(output_filename, fourcc, fps, (width, height), isColor=False)
            
            frame_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            frame_num = 0
            
            while data_grey.isOpened():
                ret, frame = data_grey.read()
                
                if ret:
                    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    adjusted_grey = cv.convertScaleAbs(grey, alpha=video_contrast, beta=video_brightness)
                    
                    out.write(adjusted_grey)
                    
                    frame_placeholder.image(adjusted_grey, caption=f"Processing {frame_num + 1}/{frame_count} Frames at {fps} FPS")
                    
                    progress = (frame_num + 1) / frame_count
                    progress_bar.progress(progress)
                    
                    frame_num += 1
                else:
                    break
            
            data_grey.release()
            out.release()
            
            if os.path.exists(output_filename):
                with open(output_filename, "rb") as file:
                    st.download_button(
                        label="Download Video",
                        data=file.read(),
                        file_name=f"{uploaded_file.name.split('.')[0]}_gray.mp4",
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
data_grey = cv2.VideoCapture('video.mp4')

while data_grey.isOpened():
        
    # Read frame from video
    ret, frame = data_grey.read()
        
    # Check if frame was read successfully
    if ret:

        # Convert frame to grayscale
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply brightness and contrast adjustments
        adjusted_grey = cv2.convertScaleAbs(grey, alpha={video_contrast}, beta={video_brightness})

        # Display the adjusted grayscale image
        cv2.imshow('Adjusted Grey', adjusted_grey)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(25) == ord('q'):
            break       
    else:
        break

# Release video capture object and close all windows
data_grey.release()
out.release()
cv2.destroyAllWindows()
        ''', language='python')
        
    else:
        st.error("Unsupported file format. Please upload an image (jpg, jpeg, png) or video (mp4, avi, mov) file.")

else:
    st.info('Processing will be auto-detected based on the file.')