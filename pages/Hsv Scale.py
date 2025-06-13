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
        st.subheader("HSV Scale")
    with col3:
        pass

with st.expander('Functions'):
    col1, col2 = st.columns(2)
    with col1:
        st.help(cv.cvtColor)
    with col2:
        st.help(cv.COLOR_BGR2HSV)  

st.divider()

uploaded_file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

st.divider()

if uploaded_file is not None:

    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension in ['jpg', 'jpeg', 'png']:
        st.subheader("Image Processing")
        
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv.imdecode(file_bytes, 1)
        
        hue_shift = st.slider('Hue Shift', min_value=-180, max_value=180, value=0, step=1)
        saturation_scale = st.slider('Saturation Scale', min_value=0.0, max_value=2.0, value=1.0, step=0.1)
        value_scale = st.slider('Value/Brightness Scale', min_value=0.0, max_value=2.0, value=1.0, step=0.1)
        
        hsv = cv.cvtColor(opencv_image.copy(), cv.COLOR_BGR2HSV)
        
        hsv_adjusted = hsv.copy().astype(np.float32)
        hsv_adjusted[:,:,0] = (hsv_adjusted[:,:,0] + hue_shift) % 180
        hsv_adjusted[:,:,1] = np.clip(hsv_adjusted[:,:,1] * saturation_scale, 0, 255)
        hsv_adjusted[:,:,2] = np.clip(hsv_adjusted[:,:,2] * value_scale, 0, 255)
        hsv_final = hsv_adjusted.astype(np.uint8)
        
        result = cv.cvtColor(hsv_final, cv.COLOR_HSV2BGR)
        st.image(result, channels="BGR")
        

        success, encoded_img = cv.imencode('.png', result)
        if success:
            img_bytes = BytesIO(encoded_img.tobytes())
            st.download_button(
                label="Download Image",
                data=img_bytes.getvalue(),
                file_name=f"{uploaded_file.name.split('.')[0]}_hsv.png",
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
        
# Convert to HSV
hsv = cv2.cvtColor(data, cv2.COLOR_BGR2HSV)

# Apply HSV adjustments
hsv_adjusted = hsv.copy().astype(np.float32)
hsv_adjusted[:,:,0] = (hsv_adjusted[:,:,0] + {hue_shift}) % 180  # Hue shift
hsv_adjusted[:,:,1] = np.clip(hsv_adjusted[:,:,1] * {saturation_scale}, 0, 255)  # Saturation
hsv_adjusted[:,:,2] = np.clip(hsv_adjusted[:,:,2] * {value_scale}, 0, 255)  # Value
hsv_final = hsv_adjusted.astype(np.uint8)

# Convert back to BGR for display
result = cv2.cvtColor(hsv_final, cv2.COLOR_HSV2BGR)

# Display the HSV adjusted image
fig, ax = plt.subplots()
ax.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
ax.set_title('HSV - H:{hue_shift}, S:{saturation_scale}, V:{value_scale}')
ax.axis('off')
plt.show()
        ''', language='python')
        
    elif file_extension in ['mp4', 'avi', 'mov']:
        st.subheader("Video Processing")
        
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
        
        video_hue_shift = st.slider('Video Hue Shift', min_value=-180, max_value=180, value=0, step=1)
        video_saturation_scale = st.slider('Video Saturation Scale', min_value=0.0, max_value=2.0, value=1.0, step=0.1)
        video_value_scale = st.slider('Video Value/Brightness Scale', min_value=0.0, max_value=2.0, value=1.0, step=0.1)

        if uploaded_file is not None:
            data_video = cv.VideoCapture('temp_video.mp4')
            
            fps = int(data_video.get(cv.CAP_PROP_FPS))
            frame_count = int(data_video.get(cv.CAP_PROP_FRAME_COUNT))
            width = int(data_video.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(data_video.get(cv.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            output_filename = f"processed_{uploaded_file.name.split('.')[0]}_hsv.mp4"
            out = cv.VideoWriter(output_filename, fourcc, fps, (width, height), isColor=True)
            
            frame_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            frame_num = 0
            
            while data_video.isOpened():
                ret, frame = data_video.read()
                
                if ret:
                    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                    
                    hsv_adjusted = hsv.copy().astype(np.float32)
                    hsv_adjusted[:,:,0] = (hsv_adjusted[:,:,0] + video_hue_shift) % 180
                    hsv_adjusted[:,:,1] = np.clip(hsv_adjusted[:,:,1] * video_saturation_scale, 0, 255)
                    hsv_adjusted[:,:,2] = np.clip(hsv_adjusted[:,:,2] * video_value_scale, 0, 255)
                    hsv_final = hsv_adjusted.astype(np.uint8)
                    
                    result = cv.cvtColor(hsv_final, cv.COLOR_HSV2BGR)
                    
                    out.write(result)
                    
                    frame_placeholder.image(result, channels="BGR", caption=f"Processing {frame_num + 1}/{frame_count} Frames at {fps} FPS")
                    
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
                        file_name=f"{uploaded_file.name.split('.')[0]}_hsv.mp4",
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

while data_video.isOpened():
        
    # Read frame from video
    ret, frame = data_video.read()
        
    # Check if frame was read successfully
    if ret:

        # Convert frame to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Apply HSV adjustments
        hsv_adjusted = hsv.copy().astype(np.float32)
        hsv_adjusted[:,:,0] = (hsv_adjusted[:,:,0] + {video_hue_shift}) % 180  # Hue shift
        hsv_adjusted[:,:,1] = np.clip(hsv_adjusted[:,:,1] * {video_saturation_scale}, 0, 255)  # Saturation
        hsv_adjusted[:,:,2] = np.clip(hsv_adjusted[:,:,2] * {video_value_scale}, 0, 255)  # Value
        hsv_final = hsv_adjusted.astype(np.uint8)

        # Convert back to BGR
        result = cv2.cvtColor(hsv_final, cv2.COLOR_HSV2BGR)

        # Display the adjusted HSV image
        cv2.imshow('Adjusted HSV', result)
        
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