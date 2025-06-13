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
        st.subheader("Laplacian Edge Detection")
    with col3:
        pass

with st.expander('Functions'):
    col1, col2 = st.columns(2)
    with col1:
        st.help(cv.Laplacian)
    with col2:
        st.help(cv.CV_32F)  

st.divider()

uploaded_file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

st.divider()

if uploaded_file is not None:

    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension in ['jpg', 'jpeg', 'png']:
        st.subheader("Image Processing")
        
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv.imdecode(file_bytes, 1)
        
        ddepth = st.slider("Depth", min_value=-1, max_value=6, value=5)
        ksize = st.slider("Kernel Size", min_value=1, max_value=31, value=3, step=2)
        
        # Depth mapping for display purposes
        depth_map = {-1: "cv2.CV_8U", 0: "cv2.CV_8U", 1: "cv2.CV_8S", 2: "cv2.CV_16U", 
                    3: "cv2.CV_16S", 4: "cv2.CV_32S", 5: "cv2.CV_32F", 6: "cv2.CV_64F"}
        depth_name = depth_map.get(ddepth, f"ddepth={ddepth}")
        
        laplacian = cv.Laplacian(opencv_image.copy(), ddepth=ddepth, ksize=ksize)
        laplacian = np.uint8(np.absolute(laplacian))
        st.image(laplacian)
        

        success, encoded_img = cv.imencode('.png', laplacian)
        if success:
            img_bytes = BytesIO(encoded_img.tobytes())
            st.download_button(
                label="Download Image",
                data=img_bytes.getvalue(),
                file_name=f"{uploaded_file.name.split('.')[0]}_laplacian.png",
                mime="image/png"
            )
        
        st.divider()

        st.subheader("Code")

        st.code(f'''
# Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image file
data = cv2.imread('image.jpg')
        
# Apply Laplacian Edge Detection
laplacian = cv2.Laplacian(data, ddepth={ddepth}, ksize={ksize})

# Convert to uint8 for display
laplacian = np.uint8(np.absolute(laplacian))

# Display the Laplacian edge detected image
fig, ax = plt.subplots()
ax.imshow(laplacian, cmap='gray')
ax.set_title('Laplacian Edge - Depth: {depth_name}, Kernel: {ksize}x{ksize}')
ax.axis('off')
plt.show()
        ''', language='python')
        
    elif file_extension in ['mp4', 'avi', 'mov']:
        st.subheader("Video Processing")
        
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
        
        video_ddepth = st.slider("Video Depth", min_value=-1, max_value=6, value=5)
        video_ksize = st.slider("Video Kernel Size", min_value=1, max_value=31, value=3, step=2)

        # Depth mapping for display purposes
        depth_map = {-1: "cv2.CV_8U", 0: "cv2.CV_8U", 1: "cv2.CV_8S", 2: "cv2.CV_16U", 
                    3: "cv2.CV_16S", 4: "cv2.CV_32S", 5: "cv2.CV_32F", 6: "cv2.CV_64F"}
        video_depth_name = depth_map.get(video_ddepth, f"ddepth={video_ddepth}")

        if uploaded_file is not None:
            data_laplacian = cv.VideoCapture('temp_video.mp4')
            
            fps = int(data_laplacian.get(cv.CAP_PROP_FPS))
            frame_count = int(data_laplacian.get(cv.CAP_PROP_FRAME_COUNT))
            width = int(data_laplacian.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(data_laplacian.get(cv.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            output_filename = f"processed_{uploaded_file.name.split('.')[0]}_laplacian.mp4"
            out = cv.VideoWriter(output_filename, fourcc, fps, (width, height), isColor=False)
            
            frame_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            frame_num = 0
            
            while data_laplacian.isOpened():
                ret, frame = data_laplacian.read()
                
                if ret:
                    laplacian_frame = cv.Laplacian(frame, ddepth=video_ddepth, ksize=video_ksize)
                    laplacian_frame = np.uint8(np.absolute(laplacian_frame))
                    
                    # Convert to grayscale for video writer compatibility
                    if len(laplacian_frame.shape) == 3:
                        laplacian_frame = cv.cvtColor(laplacian_frame, cv.COLOR_BGR2GRAY)
                    
                    out.write(laplacian_frame)
                    
                    frame_placeholder.image(laplacian_frame, caption=f"Processing {frame_num + 1}/{frame_count} Frames at {fps} FPS")
                    
                    progress = (frame_num + 1) / frame_count
                    progress_bar.progress(progress)
                    
                    frame_num += 1
                else:
                    break
            
            data_laplacian.release()
            out.release()
            
            if os.path.exists(output_filename):
                with open(output_filename, "rb") as file:
                    st.download_button(
                        label="Download Video",
                        data=file.read(),
                        file_name=f"{uploaded_file.name.split('.')[0]}_laplacian.mp4",
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
import numpy as np
        
# Load video file
data_laplacian = cv2.VideoCapture('video.mp4')

while data_laplacian.isOpened():
        
    # Read frame from video
    ret, frame = data_laplacian.read()
        
    # Check if frame was read successfully
    if ret:

        # Apply Laplacian Edge Detection
        laplacian_frame = cv2.Laplacian(frame, ddepth={video_ddepth}, ksize={video_ksize})
        
        # Convert to uint8 for display
        laplacian_frame = np.uint8(np.absolute(laplacian_frame))

        # Display the Laplacian edge detected image
        cv2.imshow('Laplacian Edge Detection', laplacian_frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(25) == ord('q'):
            break       
    else:
        break

# Release video capture object and close all windows
data_laplacian.release()
cv2.destroyAllWindows()
        ''', language='python')
        
    else:
        st.error("Unsupported file format. Please upload an image (jpg, jpeg, png) or video (mp4, avi, mov) file.")

else:
    st.info('Processing will be auto-detected based on the file.')