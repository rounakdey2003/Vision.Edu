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
        st.subheader("Text Drawing")
    with col3:
        pass

with st.expander('Functions'):
    st.help(cv.putText)

st.divider()

uploaded_file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

st.divider()

if uploaded_file is not None:

    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension in ['jpg', 'jpeg', 'png']:
        st.subheader("Image Processing")
        
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv.imdecode(file_bytes, 1)
        
        text = st.text_input("Text", value="Hello, World!")
        position_x = st.slider("Position X", min_value=0, max_value=500, value=50)
        position_y = st.slider("Position Y", min_value=0, max_value=500, value=50)
        font_scale = st.slider("Font Scale", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
        thickness = st.slider("Thickness", min_value=1, max_value=10, value=2)
        
        font_option = st.selectbox("Font", ["FONT_HERSHEY_SIMPLEX", "FONT_HERSHEY_PLAIN", "FONT_HERSHEY_DUPLEX"])
        font_map = {
            "FONT_HERSHEY_SIMPLEX": cv.FONT_HERSHEY_SIMPLEX,
            "FONT_HERSHEY_PLAIN": cv.FONT_HERSHEY_PLAIN,
            "FONT_HERSHEY_DUPLEX": cv.FONT_HERSHEY_DUPLEX
        }
        selected_font = font_map[font_option]
        
        image_copy = opencv_image.copy()
        cv.putText(image_copy, text, (position_x, position_y), selected_font, font_scale, (255, 0, 0), thickness)
        st.image(image_copy, channels="BGR")
        
        success, encoded_img = cv.imencode('.png', image_copy)
        if success:
            img_bytes = BytesIO(encoded_img.tobytes())
            st.download_button(
                label="Download Image",
                data=img_bytes.getvalue(),
                file_name=f"{uploaded_file.name.split('.')[0]}_text.png",
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

# Draw text "{text}" at position ({position_x}, {position_y})
# Font: {font_option}, Scale: {font_scale}, Thickness: {thickness}
cv2.putText(data, "{text}", ({position_x}, {position_y}), cv2.{font_option}, {font_scale}, (255, 0, 0), {thickness})

# Display the image with text
fig, ax = plt.subplots()
ax.imshow(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))
ax.set_title('Text Drawing - "{text}"')
ax.axis('off')
plt.show()
        ''', language='python')
        
    elif file_extension in ['mp4', 'avi', 'mov']:
        st.subheader("Video Processing")
        
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
        
        video_text = st.text_input("Video Text", value="Hello, World!")
        video_position_x = st.slider("Video Position X", min_value=0, max_value=500, value=50)
        video_position_y = st.slider("Video Position Y", min_value=0, max_value=500, value=50)
        video_font_scale = st.slider("Video Font Scale", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
        video_thickness = st.slider("Video Thickness", min_value=1, max_value=10, value=2)
        
        video_font_option = st.selectbox("Video Font", ["FONT_HERSHEY_SIMPLEX", "FONT_HERSHEY_PLAIN", "FONT_HERSHEY_DUPLEX"])
        font_map = {
            "FONT_HERSHEY_SIMPLEX": cv.FONT_HERSHEY_SIMPLEX,
            "FONT_HERSHEY_PLAIN": cv.FONT_HERSHEY_PLAIN,
            "FONT_HERSHEY_DUPLEX": cv.FONT_HERSHEY_DUPLEX
        }
        video_selected_font = font_map[video_font_option]

        if uploaded_file is not None:
            data_video = cv.VideoCapture('temp_video.mp4')
            
            fps = int(data_video.get(cv.CAP_PROP_FPS))
            frame_count = int(data_video.get(cv.CAP_PROP_FRAME_COUNT))
            width = int(data_video.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(data_video.get(cv.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            output_filename = f"processed_{uploaded_file.name.split('.')[0]}_text.mp4"
            out = cv.VideoWriter(output_filename, fourcc, fps, (width, height), isColor=True)
            
            frame_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            frame_num = 0
            
            while data_video.isOpened():
                ret, frame = data_video.read()
                
                if ret:
                    frame_with_text = frame.copy()
                    cv.putText(frame_with_text, video_text, (video_position_x, video_position_y), 
                              video_selected_font, video_font_scale, (255, 0, 0), video_thickness)
                    
                    out.write(frame_with_text)
                    
                    frame_placeholder.image(frame_with_text, channels="BGR", 
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
                        file_name=f"{uploaded_file.name.split('.')[0]}_text.mp4",
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

        # Draw text "{video_text}" at position ({video_position_x}, {video_position_y})
        # Font: {video_font_option}, Scale: {video_font_scale}, Thickness: {video_thickness}
        cv2.putText(frame, "{video_text}", ({video_position_x}, {video_position_y}), 
                   cv2.{video_font_option}, {video_font_scale}, (255, 0, 0), {video_thickness})

        # Display the frame with text
        cv2.imshow('Text Drawing', frame)
        
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