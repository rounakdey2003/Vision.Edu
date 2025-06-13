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
        st.subheader("Scharr Edge Detection")
    with col3:
        pass

with st.expander('Functions'):
    col1, col2 = st.columns(2)
    with col1:
        st.help(cv.Scharr)
    with col2:
        st.help(cv.CV_8U)

st.divider()

uploaded_file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

st.divider()

if uploaded_file is not None:

    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension in ['jpg', 'jpeg', 'png']:
        st.subheader("Image Processing")
        
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv.imdecode(file_bytes, 1)
        
        dx = st.slider("dx", min_value=0, max_value=2, value=1)
        dy = st.slider("dy", min_value=0, max_value=2, value=0)
        ddepth = st.slider("ddepth", min_value=-1, max_value=6, value=5)
        
        depth_map = {-1: "cv2.CV_8U", 0: "cv2.CV_8U", 1: "cv2.CV_8S", 2: "cv2.CV_16U", 
                    3: "cv2.CV_16S", 4: "cv2.CV_32S", 5: "cv2.CV_32F", 6: "cv2.CV_64F"}
        depth_name = depth_map.get(ddepth, f"ddepth={ddepth}")

        if dx == 1 and dy == 0:
            direction = "X-direction (vertical edges)"
        elif dx == 0 and dy == 1:
            direction = "Y-direction (horizontal edges)"
        elif dx == 1 and dy == 1:
            direction = "Both X and Y directions"
        else:
            direction = f"dx={dx}, dy={dy}"
        
        scharr = cv.Scharr(opencv_image.copy(), ddepth=ddepth, dx=dx, dy=dy)
        if ddepth == cv.CV_8U:
            scharr = np.uint8(scharr)
        else:
            scharr = np.uint8(np.absolute(scharr))
        st.image(scharr)

        success, encoded_img = cv.imencode('.png', scharr)
        if success:
            img_bytes = BytesIO(encoded_img.tobytes())
            st.download_button(
                label="Download Image",
                data=img_bytes.getvalue(),
                file_name=f"{uploaded_file.name.split('.')[0]}_scharr.png",
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
        
# Scharr Edge Detection: {direction}
# Parameters: dx={dx}, dy={dy}, ddepth={ddepth} ({depth_name})
scharr = cv2.Scharr(data.copy(), 
                ddepth={ddepth}, 
                dx={dx}, 
                dy={dy})

# Process output based on depth type
if ddepth == cv2.CV_8U:
    scharr = np.uint8(scharr)
else:
    scharr = np.uint8(np.absolute(scharr))

# Display the Scharr edge detected image
fig, ax = plt.subplots()
ax.imshow(scharr, cmap='gray')
ax.set_title('Scharr Edge - {direction} | Depth: {depth_name}')
ax.axis('off')
plt.show()
        ''', language='python')
        
    elif file_extension in ['mp4', 'avi', 'mov']:
        st.subheader("Video Processing")
        
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
        
        video_dx = st.slider("Video dx", min_value=0, max_value=2, value=1)
        video_dy = st.slider("Video dy", min_value=0, max_value=2, value=0)
        video_ddepth = st.slider("Video ddepth", min_value=-1, max_value=6, value=5)

        depth_map = {-1: "cv2.CV_8U", 0: "cv2.CV_8U", 1: "cv2.CV_8S", 2: "cv2.CV_16U", 
                    3: "cv2.CV_16S", 4: "cv2.CV_32S", 5: "cv2.CV_32F", 6: "cv2.CV_64F"}
        video_depth_name = depth_map.get(video_ddepth, f"ddepth={video_ddepth}")

        if video_dx == 1 and video_dy == 0:
            video_direction = "X-direction (vertical edges)"
        elif video_dx == 0 and video_dy == 1:
            video_direction = "Y-direction (horizontal edges)"
        elif video_dx == 1 and video_dy == 1:
            video_direction = "Both X and Y directions"
        else:
            video_direction = f"dx={video_dx}, dy={video_dy}"

        if uploaded_file is not None:
            data_video = cv.VideoCapture('temp_video.mp4')
            
            fps = int(data_video.get(cv.CAP_PROP_FPS))
            frame_count = int(data_video.get(cv.CAP_PROP_FRAME_COUNT))
            width = int(data_video.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(data_video.get(cv.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            output_filename = f"processed_{uploaded_file.name.split('.')[0]}_scharr.mp4"
            out = cv.VideoWriter(output_filename, fourcc, fps, (width, height), isColor=False)
            
            frame_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            frame_num = 0
            
            while data_video.isOpened():
                ret, frame = data_video.read()
                
                if ret:
                    scharr = cv.Scharr(frame, ddepth=video_ddepth, dx=video_dx, dy=video_dy)
                    if video_ddepth == cv.CV_8U:
                        scharr = np.uint8(scharr)
                    else:
                        scharr = np.uint8(np.absolute(scharr))
                    
                    out.write(scharr)
                    
                    frame_placeholder.image(scharr, caption=f"Processing {frame_num + 1}/{frame_count} Frames at {fps} FPS")
                    
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
                        file_name=f"{uploaded_file.name.split('.')[0]}_scharr.mp4",
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

        # Scharr Edge Detection: {video_direction}
        # Parameters: dx={video_dx}, dy={video_dy}, ddepth={video_ddepth} ({video_depth_name})
        scharr = cv2.Scharr(frame, 
                        ddepth={video_ddepth}, 
                        dx={video_dx}, 
                        dy={video_dy})

        # Process output based on depth type
        if ddepth == cv2.CV_8U:
            scharr = np.uint8(scharr)
        else:
            scharr = np.uint8(np.absolute(scharr))

        # Display the Scharr edge detected image
        cv2.imshow('Scharr Edge Detection', scharr)
        
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