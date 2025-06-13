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
        st.subheader("Flip")
    with col3:
        pass

with st.expander('Functions'):
    st.help(cv.flip)  

st.divider()

uploaded_file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

st.divider()

if uploaded_file is not None:

    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension in ['jpg', 'jpeg', 'png']:
        st.subheader("Image Processing")
        
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv.imdecode(file_bytes, 1)
        
        # Display original image
        st.image(opencv_image.copy(), channels="BGR", caption="Original Image")
        
        flip_code = st.selectbox("Flip Direction", 
                                options=[0, 1, -1], 
                                format_func=lambda x: {0: "Vertical", 1: "Horizontal", -1: "Both"}[x],
                                index=1)
        
        flip_direction = {0: "Vertical", 1: "Horizontal", -1: "Both"}[flip_code]
        
        flip = cv.flip(opencv_image.copy(), flip_code)
        st.image(flip, channels="BGR")
        
        # Download button for processed image
        success, encoded_img = cv.imencode('.png', flip)
        if success:
            img_bytes = BytesIO(encoded_img.tobytes())
            st.download_button(
                label="Download Image",
                data=img_bytes.getvalue(),
                file_name=f"{uploaded_file.name.split('.')[0]}_flip.png",
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
        
# Flip operation - Direction: {flip_direction} (flip_code={flip_code})
flip = cv2.flip(data, {flip_code})

# Display the flipped image
fig, ax = plt.subplots()
ax.imshow(flip)
ax.set_title('Flip - Direction: {flip_direction}')
ax.axis('off')
plt.show()
        ''', language='python')
        
    elif file_extension in ['mp4', 'avi', 'mov']:
        st.subheader("Video Processing")
        
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
        
        video_flip_code = st.selectbox("Video Flip Direction", 
                                      options=[0, 1, -1], 
                                      format_func=lambda x: {0: "Vertical", 1: "Horizontal", -1: "Both"}[x],
                                      index=1)
        
        video_flip_direction = {0: "Vertical", 1: "Horizontal", -1: "Both"}[video_flip_code]

        if uploaded_file is not None:
            data_flip = cv.VideoCapture('temp_video.mp4')
            
            fps = int(data_flip.get(cv.CAP_PROP_FPS))
            frame_count = int(data_flip.get(cv.CAP_PROP_FRAME_COUNT))
            width = int(data_flip.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(data_flip.get(cv.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            output_filename = f"processed_{uploaded_file.name.split('.')[0]}_flip.mp4"
            out = cv.VideoWriter(output_filename, fourcc, fps, (width, height), isColor=True)
            
            frame_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            frame_num = 0
            
            while data_flip.isOpened():
                ret, frame = data_flip.read()
                
                if ret:
                    flipped_frame = cv.flip(frame, video_flip_code)
                    
                    out.write(flipped_frame)
                    
                    frame_placeholder.image(flipped_frame, channels="BGR", caption=f"Processing {frame_num + 1}/{frame_count} Frames at {fps} FPS")
                    
                    progress = (frame_num + 1) / frame_count
                    progress_bar.progress(progress)
                    
                    frame_num += 1
                else:
                    break
            
            data_flip.release()
            out.release()
            
            if os.path.exists(output_filename):
                with open(output_filename, "rb") as file:
                    st.download_button(
                        label="Download Video",
                        data=file.read(),
                        file_name=f"{uploaded_file.name.split('.')[0]}_flip.mp4",
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
data_flip = cv2.VideoCapture('video.mp4')

while data_flip.isOpened():
        
    # Read frame from video
    ret, frame = data_flip.read()
        
    # Check if frame was read successfully
    if ret:

        # Flip operation - Direction: {video_flip_direction} (flip_code={video_flip_code})
        flipped_frame = cv2.flip(frame, {video_flip_code})

        # Display the flipped frame
        cv2.imshow('Flipped Frame', flipped_frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(25) == ord('q'):
            break       
    else:
        break

# Release video capture object and close all windows
data_flip.release()
out.release()
cv2.destroyAllWindows()
        ''', language='python')
        
    else:
        st.error("Unsupported file format. Please upload an image (jpg, jpeg, png) or video (mp4, avi, mov) file.")

else:
    st.info('Processing will be auto-detected based on the file.')