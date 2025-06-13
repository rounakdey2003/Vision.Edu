import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import tempfile
import os

st.set_page_config(
    page_title="Video Processing",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="expanded",
)

with st.container(height=100):
    col1, col2, col3 = st.columns(3)
    with col1:
        pass
    with col2:
        st.header("Selection")
    with col3:
        pass

with st.expander('Functions'):
    col1, col2 = st.columns(2)
    with col1:
        st.help(cv.VideoCapture)
    with col2:
        st.help(cv.imshow)

st.code('''
import cv2
import matplotlib.pyplot as plt
import numpy as np
        
data = cv2.VideoCapture('file_path/file_name.mp4')

while data.isOpened():
    ret, frame = data.read()
    if ret:
        cv2.imshow('Original', frame)
        if cv2.waitKey(25) == ord('q'):
            break
    else:
        break
data.release()
cv2.destroyAllWindows()
''', language='python') 

uploaded_file = st.file_uploader("", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.close()
    
    st.session_state.video_path = tfile.name
    
    cap = cv.VideoCapture(tfile.name)
    ret, frame = cap.read()
    if ret:
        st.image(frame, channels="BGR", caption="Video Preview (First Frame)")
    cap.release()

st.divider()

with st.container(height=100):
    col1, col2, col3 = st.columns(3)
    with col1:
        pass
    with col2:
        st.header("Colour")
    with col3:
        pass

st.subheader("Gray Scale")

with st.expander('Functions'):
    col1, col2 = st.columns(2)
    with col1:
        st.help(cv.cvtColor)
    with col2:
        st.help(cv.COLOR_BGR2GRAY)

st.code('''
data_grey = cv2.VideoCapture('file_path/file_name.mp4')

while data_grey.isOpened():
    ret, frame = data_grey.read()
    if ret:
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Gray', grey)
        if cv2.waitKey(25) == ord('q'):
            break
    else:
        break
data_grey.release()
cv2.destroyAllWindows()
''', language='python')    

if st.button('Generate Gray'):
    if 'video_path' not in st.session_state:
        st.error('Please upload a video first')
    else:
        cap = cv.VideoCapture(st.session_state.video_path)
        frames = []
        frame_count = 0
        max_frames = 10
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if ret:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                frames.append(gray)
                frame_count += 1
            else:
                break
        cap.release()
        
        if frames:
            st.write(f"Showing {len(frames)} frames in grayscale:")
            for i, frame in enumerate(frames):
                st.image(frame, caption=f"Frame {i+1} - Grayscale", width=300)

st.divider()

st.subheader("HSV Scale")

with st.expander('Functions'):
    col1, col2 = st.columns(2)
    with col1:
        st.help(cv.cvtColor)
    with col2:
        st.help(cv.COLOR_BGR2HSV)

st.code('''
data_hsv = cv2.VideoCapture('file_path/file_name.mp4')

while data_hsv.isOpened():
    ret, frame = data_hsv.read()
    if ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.imshow('HSV', hsv)
        if cv2.waitKey(25) == ord('q'):
            break
    else:
        break
data_hsv.release()
cv2.destroyAllWindows()
''', language='python')    

if st.button('Generate HSV'):
    if 'video_path' not in st.session_state:
        st.error('Please upload a video first')
    else:
        cap = cv.VideoCapture(st.session_state.video_path)
        frames = []
        frame_count = 0
        max_frames = 10
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if ret:
                hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                frames.append(hsv)
                frame_count += 1
            else:
                break
        cap.release()
        
        if frames:
            st.write(f"Showing {len(frames)} frames in HSV:")
            for i, frame in enumerate(frames):
                st.image(frame, caption=f"Frame {i+1} - HSV", width=300)

st.divider()

with st.container(height=100):
    col1, col2, col3 = st.columns(3)
    with col1:
        pass
    with col2:
        st.header("Resizing")
    with col3:
        pass

st.subheader("Dimension Resizing")

with st.expander('Functions'):
    st.help(cv.resize)

st.code('''
data_resizeDim = cv2.VideoCapture('file_path/file_name.mp4')

while data_resizeDim.isOpened():
    ret, frame = data_resizeDim.read()
    if ret:
        resizeDim = cv2.resize(frame, (500,500))
        cv2.imshow('Dimensional Resize', resizeDim)
        if cv2.waitKey(25) == ord('q'):
            break
    else:
        break
data_resizeDim.release()
cv2.destroyAllWindows()
''', language='python')    

if st.button('Generate Dimension Resize'):
    if 'video_path' not in st.session_state:
        st.error('Please upload a video first')
    else:
        cap = cv.VideoCapture(st.session_state.video_path)
        ret, frame = cap.read()
        if ret:
            resizeDim = cv.resize(frame, (500, 500))
            st.image(resizeDim, channels="BGR", caption="Resized to 500x500")
        cap.release()

st.divider()

st.subheader("Axis Resizing")

st.code('''
data_resizeAxis = cv2.VideoCapture('file_path/file_name.mp4')

while data_resizeAxis.isOpened():
    ret, frame = data_resizeAxis.read()
    if ret:
        resizeAxis = cv2.resize(frame, dsize=None, fx=1, fy=1)
        cv2.imshow('Axis Resize', resizeAxis)
        if cv2.waitKey(25) == ord('q'):
            break
    else:
        break
data_resizeAxis.release()
cv2.destroyAllWindows()
''', language='python')    

if st.button('Generate Axis Resize'):
    if 'video_path' not in st.session_state:
        st.error('Please upload a video first')
    else:
        cap = cv.VideoCapture(st.session_state.video_path)
        ret, frame = cap.read()
        if ret:
            resizeAxis = cv.resize(frame, None, fx=0.5, fy=0.5)
            st.image(resizeAxis, channels="BGR", caption="Resized with fx=0.5, fy=0.5")
        cap.release()

st.divider()

with st.container(height=150):
    col1, col2, col3 = st.columns(3)
    with col1:
        pass
    with col2:
        st.header("Transformation")
    with col3:
        pass

st.subheader("Rotate")

with st.expander('Functions'):
    st.help(cv.rotate)

st.code('''
data_rotate = cv2.VideoCapture('file_path/file_name.mp4')

while data_rotate.isOpened():
    ret, frame = data_rotate.read()
    if ret:
        rotate = cv2.rotate(frame, cv2.ROTATE_180)
        cv2.imshow('Rotate', rotate)
        if cv2.waitKey(25) == ord('q'):
            break
    else:
        break
data_rotate.release()
cv2.destroyAllWindows()
''', language='python')    

if st.button('Generate Rotate'):
    if 'video_path' not in st.session_state:
        st.error('Please upload a video first')
    else:
        cap = cv.VideoCapture(st.session_state.video_path)
        ret, frame = cap.read()
        if ret:
            rotate = cv.rotate(frame, cv.ROTATE_180)
            st.image(rotate, channels="BGR", caption="Rotated 180 degrees")
        cap.release()

st.divider()

st.subheader("Flip")

with st.expander('Functions'):
    st.help(cv.flip)

st.code('''
data_flip = cv2.VideoCapture('file_path/file_name.mp4')

while data_flip.isOpened():
    ret, frame = data_flip.read()
    if ret:
        flip = cv2.flip(frame, 1)
        cv2.imshow('Flip', flip)
        if cv2.waitKey(25) == ord('q'):
            break
    else:
        break
data_flip.release()
cv2.destroyAllWindows()
''', language='python')    

if st.button('Generate Flip'):
    if 'video_path' not in st.session_state:
        st.error('Please upload a video first')
    else:
        cap = cv.VideoCapture(st.session_state.video_path)
        ret, frame = cap.read()
        if ret:
            flip = cv.flip(frame, 1)
            st.image(flip, channels="BGR", caption="Flipped horizontally")
        cap.release()

st.divider()

st.subheader("Region Of Interest")

with st.expander('Functions'):
    st.help(cv.warpAffine)

st.code('''
data_roi = cv2.VideoCapture('file_path/file_name.mp4')

while data_roi.isOpened():
    ret, frame = data_roi.read()
    if ret:
        M = np.float32([[1,0,100],[0,1,100]])
        roi = cv2.warpAffine(frame, M, dsize=(1000, 1000))
        cv2.imshow('Region of interest', roi)
        if cv2.waitKey(25) == ord('q'):
            break
    else:
        break
data_roi.release()
cv2.destroyAllWindows()
''', language='python')    

if st.button('Generate ROI'):
    if 'video_path' not in st.session_state:
        st.error('Please upload a video first')
    else:
        cap = cv.VideoCapture(st.session_state.video_path)
        ret, frame = cap.read()
        if ret:
            M = np.float32([[1,0,100],[0,1,100]])
            roi = cv.warpAffine(frame, M, dsize=(frame.shape[1], frame.shape[0]))
            st.image(roi, channels="BGR", caption="Region of Interest")
        cap.release()

st.divider()

with st.container(height=100):
    col1, col2, col3 = st.columns(3)
    with col1:
        pass
    with col2:
        st.header("Filtering")
    with col3:
        pass

st.subheader("Blur")

with st.expander('Functions'):
    st.help(cv.blur)

st.code('''
data_blur = cv2.VideoCapture('file_path/file_name.mp4')

while data_blur.isOpened():
    ret, frame = data_blur.read()
    if ret:
        blur = cv2.blur(frame, ksize=(20,20))
        cv2.imshow('Blur', blur)
        if cv2.waitKey(25) == ord('q'):
            break
    else:
        break
data_blur.release()
cv2.destroyAllWindows()
''', language='python')    

if st.button('Generate Blur'):
    if 'video_path' not in st.session_state:
        st.error('Please upload a video first')
    else:
        cap = cv.VideoCapture(st.session_state.video_path)
        ret, frame = cap.read()
        if ret:
            blur = cv.blur(frame, ksize=(20,20))
            st.image(blur, channels="BGR", caption="Blurred")
        cap.release()

st.divider()

st.subheader("Gaussian Blur")

with st.expander('Functions'):
    st.help(cv.GaussianBlur)

st.code('''
data_gaussianBlur = cv2.VideoCapture('file_path/file_name.mp4')

while data_gaussianBlur.isOpened():
    ret, frame = data_gaussianBlur.read()
    if ret:
        gaussianBlur = cv2.GaussianBlur(frame, ksize=(19,19), sigmaX=9)
        cv2.imshow('Gaussian Blur', gaussianBlur)
        if cv2.waitKey(25) == ord('q'):
            break
    else:
        break
data_gaussianBlur.release()
cv2.destroyAllWindows()
''', language='python')    

if st.button('Generate Gaussian Blur'):
    if 'video_path' not in st.session_state:
        st.error('Please upload a video first')
    else:
        cap = cv.VideoCapture(st.session_state.video_path)
        ret, frame = cap.read()
        if ret:
            gaussianBlur = cv.GaussianBlur(frame, ksize=(19,19), sigmaX=9)
            st.image(gaussianBlur, channels="BGR", caption="Gaussian Blurred")
        cap.release()

st.divider()

st.subheader("Median Blur")

with st.expander('Functions'):
    st.help(cv.medianBlur)

st.code('''
data_medianBlur = cv2.VideoCapture('file_path/file_name.mp4')

while data_medianBlur.isOpened():
    ret, frame = data_medianBlur.read()
    if ret:
        medianBlur = cv2.medianBlur(frame, ksize=19)
        cv2.imshow('Median Blur', medianBlur)
        if cv2.waitKey(25) == ord('q'):
            break
    else:
        break
data_medianBlur.release()
cv2.destroyAllWindows()
''', language='python')    

if st.button('Generate Median Blur'):
    if 'video_path' not in st.session_state:
        st.error('Please upload a video first')
    else:
        cap = cv.VideoCapture(st.session_state.video_path)
        ret, frame = cap.read()
        if ret:
            medianBlur = cv.medianBlur(frame, ksize=19)
            st.image(medianBlur, channels="BGR", caption="Median Blurred")
        cap.release()

st.divider()

st.subheader("Noise Reduction")

with st.expander('Functions'):
    st.help(cv.fastNlMeansDenoisingColored)

st.code('''
data_deNoise = cv2.VideoCapture('file_path/file_name.mp4')

while data_deNoise.isOpened():
    ret, frame = data_deNoise.read()
    if ret:
        deNoise = cv2.fastNlMeansDenoisingColored(frame, h=10)
        cv2.imshow('De-Noise', deNoise)
        if cv2.waitKey(25) == ord('q'):
            break
    else:
        break
data_deNoise.release()
cv2.destroyAllWindows()
''', language='python')

if st.button('Generate Noise Reduction'):
    if 'video_path' not in st.session_state:
        st.error('Please upload a video first')
    else:
        cap = cv.VideoCapture(st.session_state.video_path)
        ret, frame = cap.read()
        if ret:
            deNoise = cv.fastNlMeansDenoisingColored(frame, h=10)
            st.image(deNoise, channels="BGR", caption="Noise Reduced")
        cap.release()

st.divider()

with st.container(height=150):
    col1, col2, col3 = st.columns(3)
    with col1:
        pass
    with col2:
        st.header("Edge Detection")
    with col3:
        pass

st.subheader("Canny")

with st.expander('Functions'):
    st.help(cv.Canny)

st.code('''
data_canny = cv2.VideoCapture('file_path/file_name.mp4')

while data_canny.isOpened():
    ret, frame = data_canny.read()
    if ret:
        canny = cv2.Canny(frame, threshold1=100, threshold2=200)
        cv2.imshow('Canny Edge Detection', canny)
        if cv2.waitKey(25) == ord('q'):
            break
    else:
        break
data_canny.release()
cv2.destroyAllWindows()
''', language='python')

if st.button('Generate Canny'):
    if 'video_path' not in st.session_state:
        st.error('Please upload a video first')
    else:
        cap = cv.VideoCapture(st.session_state.video_path)
        ret, frame = cap.read()
        if ret:
            canny = cv.Canny(frame, threshold1=100, threshold2=200)
            st.image(canny, caption="Canny Edge Detection")
        cap.release()

st.divider()

st.subheader("Sobel")

with st.expander('Functions'):
    st.help(cv.Sobel)

st.code('''
data_sobel = cv2.VideoCapture('file_path/file_name.mp4')

while data_sobel.isOpened():
    ret, frame = data_sobel.read()
    if ret:
        sobel = cv2.Sobel(frame, ddepth=5, dx=1, dy=1, ksize=3)
        cv2.imshow('Sobel Edge Detection', sobel)
        if cv2.waitKey(25) == ord('q'):
            break
    else:
        break
data_sobel.release()
cv2.destroyAllWindows()
''', language='python')

if st.button('Generate Sobel'):
    if 'video_path' not in st.session_state:
        st.error('Please upload a video first')
    else:
        cap = cv.VideoCapture(st.session_state.video_path)
        ret, frame = cap.read()
        if ret:
            sobel = cv.Sobel(frame, ddepth=5, dx=1, dy=1, ksize=3)
            sobel = np.uint8(sobel * 255 / np.max(sobel))
            st.image(sobel, channels="BGR", caption="Sobel Edge Detection")
        cap.release()

st.divider()

st.subheader("Scharr")

with st.expander('Functions'):
    st.help(cv.Scharr)

st.code('''
data_scharr = cv2.VideoCapture('file_path/file_name.mp4')

while data_scharr.isOpened():
    ret, frame = data_scharr.read()
    if ret:
        scharr = cv2.Scharr(frame, ddepth=cv2.CV_64F, dx=1, dy=0)
        cv2.imshow('Scharr Edge Detection', scharr)
        if cv2.waitKey(25) == ord('q'):
            break
    else:
        break
data_scharr.release()
cv2.destroyAllWindows()
''', language='python')

if st.button('Generate Scharr'):
    if 'video_path' not in st.session_state:
        st.error('Please upload a video first')
    else:
        cap = cv.VideoCapture(st.session_state.video_path)
        ret, frame = cap.read()
        if ret:
            scharr = cv.Scharr(frame, ddepth=cv.CV_64F, dx=1, dy=0)
            scharr = np.uint8(np.absolute(scharr))
            st.image(scharr, channels="BGR", caption="Scharr Edge Detection")
        cap.release()

st.divider()

st.subheader("Laplacian")

with st.expander('Functions'):
    st.help(cv.Laplacian)

st.code('''
data_laplacian = cv2.VideoCapture('file_path/file_name.mp4')

while data_laplacian.isOpened():
    ret, frame = data_laplacian.read()
    if ret:
        laplacian = cv2.Laplacian(frame, ddepth=cv2.CV_64F)
        cv2.imshow('Laplacian Edge Detection', laplacian)
        if cv2.waitKey(25) == ord('q'):
            break
    else:
        break
data_laplacian.release()
cv2.destroyAllWindows()
''', language='python')

if st.button('Generate Laplacian'):
    if 'video_path' not in st.session_state:
        st.error('Please upload a video first')
    else:
        cap = cv.VideoCapture(st.session_state.video_path)
        ret, frame = cap.read()
        if ret:
            laplacian = cv.Laplacian(frame, ddepth=cv.CV_64F)
            laplacian = np.uint8(np.absolute(laplacian))
            st.image(laplacian, channels="BGR", caption="Laplacian Edge Detection")
        cap.release()

st.divider()

st.subheader("Erode")

with st.expander('Functions'):
    st.help(cv.erode)

st.code('''
data_erode = cv2.VideoCapture('file_path/file_name.mp4')

while data_erode.isOpened():
    ret, frame = data_erode.read()
    if ret:
        kernel = np.ones((5,5), np.uint8)
        erode = cv2.erode(frame, kernel, iterations=1)
        cv2.imshow('Erode', erode)
        if cv2.waitKey(25) == ord('q'):
            break
    else:
        break
data_erode.release()
cv2.destroyAllWindows()
''', language='python')

if st.button('Generate Erode'):
    if 'video_path' not in st.session_state:
        st.error('Please upload a video first')
    else:
        cap = cv.VideoCapture(st.session_state.video_path)
        ret, frame = cap.read()
        if ret:
            kernel = np.ones((5,5), np.uint8)
            erode = cv.erode(frame, kernel, iterations=1)
            st.image(erode, channels="BGR", caption="Eroded Image")
        cap.release()

st.divider()

st.subheader("Dilate")

with st.expander('Functions'):
    st.help(cv.dilate)

st.code('''
data_dilate = cv2.VideoCapture('file_path/file_name.mp4')

while data_dilate.isOpened():
    ret, frame = data_dilate.read()
    if ret:
        kernel = np.ones((5,5), np.uint8)
        dilate = cv2.dilate(frame, kernel, iterations=1)
        cv2.imshow('Dilate', dilate)
        if cv2.waitKey(25) == ord('q'):
            break
    else:
        break
data_dilate.release()
cv2.destroyAllWindows()
''', language='python')

if st.button('Generate Dilate'):
    if 'video_path' not in st.session_state:
        st.error('Please upload a video first')
    else:
        cap = cv.VideoCapture(st.session_state.video_path)
        ret, frame = cap.read()
        if ret:
            kernel = np.ones((5,5), np.uint8)
            dilate = cv.dilate(frame, kernel, iterations=1)
            st.image(dilate, channels="BGR", caption="Dilated Image")
        cap.release()

st.divider()

st.subheader("Contour Detection")

with st.expander('Functions'):
    col1, col2 = st.columns(2)
    with col1:
        st.help(cv.findContours)
    with col2:
        st.help(cv.drawContours)

st.code('''
data_contour = cv2.VideoCapture('file_path/file_name.mp4')

while data_contour.isOpened():
    ret, frame = data_contour.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = cv2.drawContours(frame.copy(), contours, -1, (0, 255, 0), 2)
        cv2.imshow('Contours', contour_img)
        if cv2.waitKey(25) == ord('q'):
            break
    else:
        break
data_contour.release()
cv2.destroyAllWindows()
''', language='python')

if st.button('Generate Contours'):
    if 'video_path' not in st.session_state:
        st.error('Please upload a video first')
    else:
        cap = cv.VideoCapture(st.session_state.video_path)
        ret, frame = cap.read()
        if ret:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            _, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
            contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            contour_img = cv.drawContours(frame.copy(), contours, -1, (0, 255, 0), 2)
            st.image(contour_img, channels="BGR", caption=f"Contours Detected: {len(contours)}")
        cap.release()

st.divider()

with st.container(height=100):
    col1, col2, col3 = st.columns(3)
    with col1:
        pass
    with col2:
        st.header("Drawings")
    with col3:
        pass

st.subheader("Line")

with st.expander('Functions'):
    st.help(cv.line)

st.code('''
data_line = cv2.VideoCapture('file_path/file_name.mp4')

while data_line.isOpened():
    ret, frame = data_line.read()
    if ret:
        line = cv2.line(frame, pt1=(500, 500), pt2=(1000, 1000), color=(0, 255, 0), thickness=5)
        cv2.imshow('Line', line)
        if cv2.waitKey(25) == ord('q'):
            break
    else:
        break
data_line.release()
cv2.destroyAllWindows()
''', language='python')

if st.button('Generate Line'):
    if 'video_path' not in st.session_state:
        st.error('Please upload a video first')
    else:
        cap = cv.VideoCapture(st.session_state.video_path)
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            line = cv.line(frame, pt1=(50, 50), pt2=(min(w-50, 200), min(h-50, 150)), color=(0, 255, 0), thickness=5)
            st.image(line, channels="BGR", caption="Line Drawing")
        cap.release()

st.divider()

st.subheader("Rectangle")

with st.expander('Functions'):
    st.help(cv.rectangle)

st.code('''
data_rectangle = cv2.VideoCapture('file_path/file_name.mp4')

while data_rectangle.isOpened():
    ret, frame = data_rectangle.read()
    if ret:
        rectangle = cv2.rectangle(frame, pt1=(500, 500), pt2=(1000, 1000), color=(0, 255, 0), thickness=5)
        cv2.imshow('Rectangle', rectangle)
        if cv2.waitKey(25) == ord('q'):
            break
    else:
        break
data_rectangle.release()
cv2.destroyAllWindows()
''', language='python')

if st.button('Generate Rectangle'):
    if 'video_path' not in st.session_state:
        st.error('Please upload a video first')
    else:
        cap = cv.VideoCapture(st.session_state.video_path)
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            rectangle = cv.rectangle(frame, pt1=(50, 50), pt2=(min(w-50, 200), min(h-50, 150)), color=(0, 255, 0), thickness=5)
            st.image(rectangle, channels="BGR", caption="Rectangle Drawing")
        cap.release()

st.divider()

st.subheader("Circle")

with st.expander('Functions'):
    st.help(cv.circle)

st.code('''
data_circle = cv2.VideoCapture('file_path/file_name.mp4')

while data_circle.isOpened():
    ret, frame = data_circle.read()
    if ret:
        circle = cv2.circle(frame, center=(1000,500), radius=100, color=(0,255,0), thickness=5)
        cv2.imshow('Circle', circle)
        if cv2.waitKey(25) == ord('q'):
            break
    else:
        break
data_circle.release()
cv2.destroyAllWindows()
''', language='python')

if st.button('Generate Circle'):
    if 'video_path' not in st.session_state:
        st.error('Please upload a video first')
    else:
        cap = cv.VideoCapture(st.session_state.video_path)
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            circle = cv.circle(frame, center=(w//2, h//2), radius=min(w, h)//10, color=(0,255,0), thickness=5)
            st.image(circle, channels="BGR", caption="Circle Drawing")
        cap.release()

st.divider()

st.subheader("Text Drawing")

with st.expander('Functions'):
    st.help(cv.putText)

st.code('''
data_text = cv2.VideoCapture('file_path/file_name.mp4')

while data_text.isOpened():
    ret, frame = data_text.read()
    if ret:
        text = cv2.putText(frame, text='hello', org=(1000,500), fontFace=4, fontScale=5, color=(0,255,0))
        cv2.imshow('Text', text)
        if cv2.waitKey(25) == ord('q'):
            break
    else:
        break
data_text.release()
cv2.destroyAllWindows()
''', language='python')

if st.button('Generate Text'):
    if 'video_path' not in st.session_state:
        st.error('Please upload a video first')
    else:
        cap = cv.VideoCapture(st.session_state.video_path)
        ret, frame = cap.read()
        if ret:
            text = cv.putText(frame, text='Hello World', org=(50, 50), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0,255,0), thickness=3)
            st.image(text, channels="BGR", caption="Text Drawing")
        cap.release()

if 'video_path' in st.session_state:
    try:
        if os.path.exists(st.session_state.video_path):
            pass
    except:
        pass