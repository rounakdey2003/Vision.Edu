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
        st.subheader("Dilate")
    with col3:
        pass

with st.expander('Functions'):
    col1, col2 = st.columns(2)
    with col1:
        st.help(cv.dilate)
    with col2:
        st.help(cv.getStructuringElement)

st.divider()

uploaded_file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

st.divider()

if uploaded_file is not None:

    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension in ['jpg', 'jpeg', 'png']:
        st.subheader("Image Processing")
        
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv.imdecode(file_bytes, 1)
        
        kernel_size = st.slider('Kernel Size', min_value=1, max_value=15, value=5, step=2)
        iterations = st.slider('Iterations', min_value=1, max_value=5, value=1, step=1)
        
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_image = cv.dilate(opencv_image.copy(), kernel, iterations=iterations)
        st.image(dilated_image, channels="BGR")
        

        success, encoded_img = cv.imencode('.png', dilated_image)
        if success:
            img_bytes = BytesIO(encoded_img.tobytes())
            st.download_button(
                label="Download Image",
                data=img_bytes.getvalue(),
                file_name=f"{uploaded_file.name.split('.')[0]}_dilated.png",
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
        
# Create structuring element (kernel)
kernel = np.ones(({kernel_size}, {kernel_size}), np.uint8)

# Apply dilate operation
dilated_image = cv2.dilate(data.copy(), kernel, iterations={iterations})

# Display the dilated image
fig, ax = plt.subplots()
ax.imshow(cv2.cvtColor(dilated_image, cv2.COLOR_BGR2RGB))
ax.set_title('Dilate - Kernel: {kernel_size}x{kernel_size}, Iterations: {iterations}')
ax.axis('off')
plt.show()
        ''', language='python')
        
    elif file_extension in ['mp4', 'avi', 'mov']:
        st.subheader("Video Processing")
        
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
        
        video_kernel_size = st.slider('Video Kernel Size', min_value=1, max_value=15, value=5, step=2)
        video_iterations = st.slider('Video Iterations', min_value=1, max_value=5, value=1, step=1)

        if uploaded_file is not None:
            data_video = cv.VideoCapture('temp_video.mp4')
            
            fps = int(data_video.get(cv.CAP_PROP_FPS))
            frame_count = int(data_video.get(cv.CAP_PROP_FRAME_COUNT))
            width = int(data_video.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(data_video.get(cv.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            output_filename = f"processed_{uploaded_file.name.split('.')[0]}_dilated.mp4"
            out = cv.VideoWriter(output_filename, fourcc, fps, (width, height), isColor=True)
            
            frame_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            frame_num = 0
            kernel = np.ones((video_kernel_size, video_kernel_size), np.uint8)
            
            while data_video.isOpened():
                ret, frame = data_video.read()
                
                if ret:
                    dilated_frame = cv.dilate(frame, kernel, iterations=video_iterations)
                    
                    out.write(dilated_frame)
                    
                    frame_placeholder.image(dilated_frame, channels="BGR", caption=f"Processing {frame_num + 1}/{frame_count} Frames at {fps} FPS")
                    
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
                        file_name=f"{uploaded_file.name.split('.')[0]}_dilated.mp4",
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
data_video = cv2.VideoCapture('video.mp4')

# Create structuring element (kernel)
kernel = np.ones(({video_kernel_size}, {video_kernel_size}), np.uint8)

while data_video.isOpened():
        
    # Read frame from video
    ret, frame = data_video.read()
        
    # Check if frame was read successfully
    if ret:

        # Apply dilate operation
        dilated_frame = cv2.dilate(frame, kernel, iterations={video_iterations})

        # Display the dilated frame
        cv2.imshow('Dilated Frame', dilated_frame)
        
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