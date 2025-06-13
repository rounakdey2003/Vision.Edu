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
        st.subheader("Region Of Interest")
    with col3:
        pass

with st.expander('Functions'):
    col1, col2 = st.columns(2)
    with col1:
        st.help(cv.warpAffine)
    with col2:
        st.help(np.float32)

st.divider()

uploaded_file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

st.divider()

if uploaded_file is not None:

    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension in ['jpg', 'jpeg', 'png']:
        st.subheader("Image Processing")
        
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv.imdecode(file_bytes, 1)
        
        x_offset = st.slider("X Offset", min_value=-50, max_value=50, value=10)
        y_offset = st.slider("Y Offset", min_value=-50, max_value=50, value=10)
        width = st.slider("Width", min_value=50, max_value=300, value=100)
        height = st.slider("Height", min_value=50, max_value=300, value=100)
        
        M = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
        roi = cv.warpAffine(opencv_image.copy(), M, dsize=(width, height))
        st.image(roi, channels="BGR")
        
        success, encoded_img = cv.imencode('.png', roi)
        if success:
            img_bytes = BytesIO(encoded_img.tobytes())
            st.download_button(
                label="Download Image",
                data=img_bytes.getvalue(),
                file_name=f"{uploaded_file.name.split('.')[0]}_roi.png",
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

# Region of Interest with offset=({x_offset},{y_offset}), size=({width},{height})
# Create transformation matrix for translation
M = np.float32([[1, 0, {x_offset}], 
                [0, 1, {y_offset}]])

# Apply warp affine transformation to extract ROI
roi = cv2.warpAffine(data.copy(), M, dsize=({width}, {height}))

# Display the ROI
fig, ax = plt.subplots()
ax.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
ax.set_title('ROI - Offset: ({x_offset},{y_offset}), Size: {width}x{height}')
ax.axis('off')
plt.show()
        ''', language='python')
        
    elif file_extension in ['mp4', 'avi', 'mov']:
        st.subheader("Video Processing")
        
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
        
        video_x_offset = st.slider("Video X Offset", min_value=-50, max_value=50, value=10)
        video_y_offset = st.slider("Video Y Offset", min_value=-50, max_value=50, value=10)
        video_width = st.slider("Video Width", min_value=50, max_value=300, value=100)
        video_height = st.slider("Video Height", min_value=50, max_value=300, value=100)

        if uploaded_file is not None:
            data_roi = cv.VideoCapture('temp_video.mp4')
            
            fps = int(data_roi.get(cv.CAP_PROP_FPS))
            frame_count = int(data_roi.get(cv.CAP_PROP_FRAME_COUNT))
            
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            output_filename = f"processed_{uploaded_file.name.split('.')[0]}_roi.mp4"
            out = cv.VideoWriter(output_filename, fourcc, fps, (video_width, video_height), isColor=True)
            
            frame_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            frame_num = 0
            
            while data_roi.isOpened():
                ret, frame = data_roi.read()
                
                if ret:
                    M = np.float32([[1, 0, video_x_offset], [0, 1, video_y_offset]])
                    roi_frame = cv.warpAffine(frame, M, dsize=(video_width, video_height))
                    
                    out.write(roi_frame)
                    
                    frame_placeholder.image(roi_frame, channels="BGR", caption=f"Processing {frame_num + 1}/{frame_count} Frames at {fps} FPS")
                    
                    progress = (frame_num + 1) / frame_count
                    progress_bar.progress(progress)
                    
                    frame_num += 1
                else:
                    break
            
            data_roi.release()
            out.release()
            
            if os.path.exists(output_filename):
                with open(output_filename, "rb") as file:
                    st.download_button(
                        label="Download Video",
                        data=file.read(),
                        file_name=f"{uploaded_file.name.split('.')[0]}_roi.mp4",
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
data_roi = cv2.VideoCapture('video.mp4')

while data_roi.isOpened():
        
    # Read frame from video
    ret, frame = data_roi.read()
        
    # Check if frame was read successfully
    if ret:

        # Create transformation matrix for translation
        M = np.float32([[1, 0, {video_x_offset}], 
                        [0, 1, {video_y_offset}]])

        # Apply warp affine transformation to extract ROI
        roi_frame = cv2.warpAffine(frame, M, dsize=({video_width}, {video_height}))

        # Display the ROI frame
        cv2.imshow('ROI Frame', roi_frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(25) == ord('q'):
            break       
    else:
        break

# Release video capture object and close all windows
data_roi.release()
out.release()
cv2.destroyAllWindows()
        ''', language='python')
        
    else:
        st.error("Unsupported file format. Please upload an image (jpg, jpeg, png) or video (mp4, avi, mov) file.")

else:
    st.info('Processing will be auto-detected based on the file.')