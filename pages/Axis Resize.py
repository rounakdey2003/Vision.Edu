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
        st.subheader("Axis Resizing")
    with col3:
        pass

with st.expander('Functions'):
    col1, col2 = st.columns(2)
    with col1:
        st.help(cv.resize)
    with col2:
        st.help(cv.INTER_LINEAR)

st.divider()

uploaded_file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

st.divider()

if uploaded_file is not None:

    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension in ['jpg', 'jpeg', 'png']:
        st.subheader("Image Processing")
        
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv.imdecode(file_bytes, 1)
        
        fx = st.slider("X-axis scaling factor", min_value=0.1, max_value=3.0, value=0.5, step=0.1)
        fy = st.slider("Y-axis scaling factor", min_value=0.1, max_value=3.0, value=0.5, step=0.1)
        
        resizeAxis = cv.resize(opencv_image.copy(), None, fx=fx, fy=fy)
        st.image(resizeAxis, channels="BGR")
        
        # Convert BGR to RGB for saving
        resizeAxis_rgb = cv.cvtColor(resizeAxis, cv.COLOR_BGR2RGB)
        success, encoded_img = cv.imencode('.png', resizeAxis_rgb)
        if success:
            img_bytes = BytesIO(encoded_img.tobytes())
            st.download_button(
                label="Download Image",
                data=img_bytes.getvalue(),
                file_name=f"{uploaded_file.name.split('.')[0]}_resized.png",
                mime="image/png"
            )
        
        st.divider()

        st.subheader("Code")

        st.code(f'''
# Import necessary libraries
import cv2
import matplotlib.pyplot as plt

# Load image file
opencv_image = cv2.imread('image.jpg')
        
# Axis resize with scaling factors: fx={fx}, fy={fy}
resizeAxis = cv2.resize(opencv_image.copy(), None, fx={fx}, fy={fy})

# Display the resized image
fig, ax = plt.subplots()
ax.imshow(cv2.cvtColor(resizeAxis, cv2.COLOR_BGR2RGB))
ax.set_title('Axis Resize - Scale: {fx}x{fy}')
ax.axis('off')
plt.show()
        ''', language='python')
        
    elif file_extension in ['mp4', 'avi', 'mov']:
        st.subheader("Video Processing")
        
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
        
        video_fx = st.slider('Video X-axis scaling factor', min_value=0.1, max_value=3.0, value=0.5, step=0.1)
        video_fy = st.slider('Video Y-axis scaling factor', min_value=0.1, max_value=3.0, value=0.5, step=0.1)

        if uploaded_file is not None:
            data_video = cv.VideoCapture('temp_video.mp4')
            
            fps = int(data_video.get(cv.CAP_PROP_FPS))
            frame_count = int(data_video.get(cv.CAP_PROP_FRAME_COUNT))
            width = int(data_video.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(data_video.get(cv.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate new dimensions
            new_width = int(width * video_fx)
            new_height = int(height * video_fy)
            
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            output_filename = f"processed_{uploaded_file.name.split('.')[0]}_resized.mp4"
            out = cv.VideoWriter(output_filename, fourcc, fps, (new_width, new_height))
            
            frame_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            frame_num = 0
            
            while data_video.isOpened():
                ret, frame = data_video.read()
                
                if ret:
                    resized_frame = cv.resize(frame, None, fx=video_fx, fy=video_fy)
                    
                    out.write(resized_frame)
                    
                    frame_placeholder.image(resized_frame, channels="BGR", caption=f"Processing {frame_num + 1}/{frame_count} Frames at {fps} FPS")
                    
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
                        file_name=f"{uploaded_file.name.split('.')[0]}_resized.mp4",
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

        # Resize frame with scaling factors: fx={video_fx}, fy={video_fy}
        resized_frame = cv2.resize(frame, None, fx={video_fx}, fy={video_fy})

        # Display the resized frame
        cv2.imshow('Resized Frame', resized_frame)
        
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