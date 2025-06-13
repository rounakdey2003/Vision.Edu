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
        st.subheader("Line Drawing")
    with col3:
        pass

with st.expander('Functions'):
    st.help(cv.line)

st.divider()

uploaded_file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

st.divider()

if uploaded_file is not None:

    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension in ['jpg', 'jpeg', 'png']:
        st.subheader("Image Processing")
        
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv.imdecode(file_bytes, 1)
        
        pt1_x = st.slider("Start X", min_value=0, max_value=500, value=50)
        pt1_y = st.slider("Start Y", min_value=0, max_value=500, value=50)
        pt2_x = st.slider("End X", min_value=0, max_value=500, value=200)
        pt2_y = st.slider("End Y", min_value=0, max_value=500, value=150)
        thickness = st.slider("Thickness", min_value=1, max_value=10, value=2)

        color_option = st.selectbox("Color", ["Green", "Red", "Blue", "White", "Black"])
        color_map = {
            "Green": (0, 255, 0),
            "Red": (0, 0, 255),
            "Blue": (255, 0, 0),
            "White": (255, 255, 255),
            "Black": (0, 0, 0)
        }
        
        selected_color = color_map[color_option]
        line_image = cv.line(opencv_image.copy(), pt1=(pt1_x, pt1_y), pt2=(pt2_x, pt2_y), 
                           color=selected_color, thickness=thickness)
        st.image(line_image, channels="BGR")
        
        # Download functionality
        success, encoded_img = cv.imencode('.png', line_image)
        if success:
            img_bytes = BytesIO(encoded_img.tobytes())
            st.download_button(
                label="Download Image",
                data=img_bytes.getvalue(),
                file_name=f"{uploaded_file.name.split('.')[0]}_line.png",
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
        
# Draw line from ({pt1_x},{pt1_y}) to ({pt2_x},{pt2_y})
# Color: {color_option} {selected_color}, Thickness: {thickness}
line = cv2.line(data.copy(), 
            pt1=({pt1_x}, {pt1_y}), 
            pt2=({pt2_x}, {pt2_y}), 
            color={selected_color},  # {color_option} in BGR
            thickness={thickness})

# Display the image with line
fig, ax = plt.subplots()
ax.imshow(line)
ax.set_title('Line: ({pt1_x},{pt1_y}) → ({pt2_x},{pt2_y}) | {color_option} | T={thickness}')
ax.axis('off')
plt.show()
        ''', language='python')
        
    elif file_extension in ['mp4', 'avi', 'mov']:
        st.subheader("Video Processing")
        
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
        
        video_pt1_x = st.slider("Video Start X", min_value=0, max_value=500, value=50)
        video_pt1_y = st.slider("Video Start Y", min_value=0, max_value=500, value=50)
        video_pt2_x = st.slider("Video End X", min_value=0, max_value=500, value=200)
        video_pt2_y = st.slider("Video End Y", min_value=0, max_value=500, value=150)
        video_thickness = st.slider("Video Thickness", min_value=1, max_value=10, value=2)

        video_color_option = st.selectbox("Video Color", ["Green", "Red", "Blue", "White", "Black"])
        video_color_map = {
            "Green": (0, 255, 0),
            "Red": (0, 0, 255),
            "Blue": (255, 0, 0),
            "White": (255, 255, 255),
            "Black": (0, 0, 0)
        }
        
        video_selected_color = video_color_map[video_color_option]

        if uploaded_file is not None:
            data_video = cv.VideoCapture('temp_video.mp4')
            
            fps = int(data_video.get(cv.CAP_PROP_FPS))
            frame_count = int(data_video.get(cv.CAP_PROP_FRAME_COUNT))
            width = int(data_video.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(data_video.get(cv.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            output_filename = f"processed_{uploaded_file.name.split('.')[0]}_line.mp4"
            out = cv.VideoWriter(output_filename, fourcc, fps, (width, height))
            
            frame_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            frame_num = 0
            
            while data_video.isOpened():
                ret, frame = data_video.read()
                
                if ret:
                    line_frame = cv.line(frame.copy(), pt1=(video_pt1_x, video_pt1_y), 
                                       pt2=(video_pt2_x, video_pt2_y), 
                                       color=video_selected_color, thickness=video_thickness)
                    
                    out.write(line_frame)
                    
                    frame_placeholder.image(line_frame, channels="BGR", 
                                          caption=f"Processing {frame_num + 1}/{frame_count} Frames at {fps} FPS")
                    
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
                        file_name=f"{uploaded_file.name.split('.')[0]}_line.mp4",
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

        # Draw line on frame
        line_frame = cv2.line(frame.copy(), 
                            pt1=({video_pt1_x}, {video_pt1_y}), 
                            pt2=({video_pt2_x}, {video_pt2_y}), 
                            color={video_selected_color},  # {video_color_option} in BGR
                            thickness={video_thickness})

        # Display the frame with line
        cv2.imshow('Line Video', line_frame)
        
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