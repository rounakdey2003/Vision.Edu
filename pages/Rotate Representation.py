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
        st.subheader("Rotate")
    with col3:
        pass

with st.expander('Functions'):
    st.help(cv.rotate)

st.divider()

uploaded_file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

st.divider()

if uploaded_file is not None:

    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension in ['jpg', 'jpeg', 'png']:
        st.subheader("Image Processing")
        
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv.imdecode(file_bytes, 1)
        
        rotation_option = st.selectbox("Rotation", ["90° Clockwise", "90° Counter-Clockwise", "180°"])

        rotation_map = {
            "90° Clockwise": cv.ROTATE_90_CLOCKWISE,
            "90° Counter-Clockwise": cv.ROTATE_90_COUNTERCLOCKWISE,
            "180°": cv.ROTATE_180
        }

        rotation_constant_map = {
            "90° Clockwise": "cv2.ROTATE_90_CLOCKWISE",
            "90° Counter-Clockwise": "cv2.ROTATE_90_COUNTERCLOCKWISE", 
            "180°": "cv2.ROTATE_180"
        }
        
        selected_constant = rotation_constant_map[rotation_option]
        rotate = cv.rotate(opencv_image.copy(), rotation_map[rotation_option])
        st.image(rotate, channels="BGR")

        success, encoded_img = cv.imencode('.png', rotate)
        if success:
            img_bytes = BytesIO(encoded_img.tobytes())
            st.download_button(
                label="Download Image",
                data=img_bytes.getvalue(),
                file_name=f"{uploaded_file.name.split('.')[0]}_rotated.png",
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
        
# Rotate image: {rotation_option}
rotate = cv2.rotate(data.copy(), {selected_constant})

# Display the rotated image
fig, ax = plt.subplots()
ax.imshow(rotate)
ax.set_title('Rotated Image - {rotation_option}')
ax.axis('off')
plt.show()
        ''', language='python')
        
    elif file_extension in ['mp4', 'avi', 'mov']:
        st.subheader("Video Processing")
        
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
        
        video_rotation_option = st.selectbox("Video Rotation", ["90° Clockwise", "90° Counter-Clockwise", "180°"])

        video_rotation_map = {
            "90° Clockwise": cv.ROTATE_90_CLOCKWISE,
            "90° Counter-Clockwise": cv.ROTATE_90_COUNTERCLOCKWISE,
            "180°": cv.ROTATE_180
        }

        video_rotation_constant_map = {
            "90° Clockwise": "cv2.ROTATE_90_CLOCKWISE",
            "90° Counter-Clockwise": "cv2.ROTATE_90_COUNTERCLOCKWISE", 
            "180°": "cv2.ROTATE_180"
        }

        if uploaded_file is not None:
            data_video = cv.VideoCapture('temp_video.mp4')
            
            fps = int(data_video.get(cv.CAP_PROP_FPS))
            frame_count = int(data_video.get(cv.CAP_PROP_FRAME_COUNT))
            width = int(data_video.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(data_video.get(cv.CAP_PROP_FRAME_HEIGHT))
            
            # Adjust dimensions based on rotation
            if video_rotation_option in ["90° Clockwise", "90° Counter-Clockwise"]:
                output_width, output_height = height, width
            else:
                output_width, output_height = width, height
            
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            output_filename = f"processed_{uploaded_file.name.split('.')[0]}_rotated.mp4"
            out = cv.VideoWriter(output_filename, fourcc, fps, (output_width, output_height))
            
            frame_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            frame_num = 0
            
            while data_video.isOpened():
                ret, frame = data_video.read()
                
                if ret:
                    rotated_frame = cv.rotate(frame, video_rotation_map[video_rotation_option])
                    
                    out.write(rotated_frame)
                    
                    frame_placeholder.image(rotated_frame, channels="BGR", caption=f"Processing {frame_num + 1}/{frame_count} Frames at {fps} FPS")
                    
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
                        file_name=f"{uploaded_file.name.split('.')[0]}_rotated.mp4",
                        mime="video/mp4"
                    )

                os.remove(output_filename)
        
        if os.path.exists("temp_video.mp4"):
            os.remove("temp_video.mp4")
        
        st.divider()

        st.subheader("Code")

        selected_video_constant = video_rotation_constant_map[video_rotation_option]

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

        # Rotate frame: {video_rotation_option}
        rotated_frame = cv2.rotate(frame, {selected_video_constant})

        # Display the rotated frame
        cv2.imshow('Rotated Frame', rotated_frame)
        
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