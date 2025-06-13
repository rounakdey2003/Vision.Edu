import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

st.set_page_config(
    page_title="Image Processing",
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
        st.help(cv.imread)
    with col2:
        st.help(cv.imshow)

st.code('''
import cv2
import matplotlib.pyplot as plt
import numpy as np

data = cv2.imread('file_path/file_name.jpg')

fig, ax = plt.subplots()
ax.imshow(data)
ax.set_title('Original')
''', language='python') 

uploaded_file = st.file_uploader("", type="jpg")
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv.imdecode(file_bytes, 1)
    st.image(opencv_image.copy(), channels="BGR")

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
grey = cv2.cvtColor(data.copy(), cv2.COLOR_BGR2GRAY)

fig, ax = plt.subplots()
ax.imshow(grey)
ax.set_title('Gray Scale')
''', language='python')    

if st.button('Generate Gray'):
    if 'opencv_image' not in locals():
        st.error('Please upload an image first')
    else:
        gray = cv.cvtColor(opencv_image.copy(), cv.COLOR_BGR2GRAY)
        st.image(gray)

st.divider()

st.subheader("HSV Scale")

with st.expander('Functions'):
    col1, col2 = st.columns(2)
    with col1:
        st.help(cv.cvtColor)
    with col2:
        st.help(cv.COLOR_BGR2HSV)

st.code('''
hsv = cv2.cvtColor(data.copy(), cv2.COLOR_BGR2HSV)

fig , ax = plt.subplots()
ax.imshow(hsv)
ax.set_title('HSV Scale')
''', language='python')    

if st.button('Generate HSV'):
    if 'opencv_image' not in locals():
        st.error('Please upload an image first')
    else:
        hsv = cv.cvtColor(opencv_image.copy(), cv.COLOR_BGR2HSV)
        st.image(hsv)

st.divider()

st.subheader("Comparison")

with st.expander('Functions'):
    col1, col2 = st.columns(2)
    with col1:
        st.help(plt.subplots)
    with col2:
        st.help(plt.axes)

st.code('''
fig, axes = plt.subplots(1, 3, figsize=(10,10))
axes[0].imshow(data)
axes[0].set_title('Original')

axes[1].imshow(grey)
axes[1].set_title('Gray Scale')

axes[2].imshow(hsv)
axes[2].set_title('HSV Scale')
''', language='python')    

if st.button('Generate All'):
    if 'opencv_image' not in locals():
        st.error('Please upload an image first')
    else:
        gray = cv.cvtColor(opencv_image.copy(), cv.COLOR_BGR2GRAY)
        hsv = cv.cvtColor(opencv_image.copy(), cv.COLOR_BGR2HSV)
        fig, axes = plt.subplots(1, 3, figsize=(10,10))

        axes[0].imshow(opencv_image.copy())
        axes[0].set_title('Original')
        axes[1].imshow(gray, cmap='gray')
        axes[1].set_title('Gray Scale')
        axes[2].imshow(hsv)
        axes[2].set_title('HSV Scale')
        st.pyplot(fig)

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
resizeDim = cv2.resize(data.copy(), (100,100))

fig, ax = plt.subplots()
ax.imshow(resizeDim)
ax.set_title('Dimension Resize')
''', language='python')    

if st.button('Generate Dimension Resize'):
    if 'opencv_image' not in locals():
        st.error('Please upload an image first')
    else:
        resizeDim = cv.resize(opencv_image.copy(), (100,100))
        st.image(resizeDim)

st.divider()

st.subheader("Axis Resizing")

with st.expander('Functions'):
    st.help(cv.resize)

st.code('''
resizeAxis = cv2.resize(data.copy(), None, fx=0.5, fy=0.5)

fig, ax = plt.subplots()
ax.imshow(resizeAxis)
ax.set_title('Axis Resize')
''', language='python')    

if st.button('Generate Axis Resize'):
    if 'opencv_image' not in locals():
        st.error('Please upload an image first')
    else:
        resizeAxis = cv.resize(opencv_image.copy(), None, fx=0.5, fy=0.5)
        st.image(resizeAxis)

st.divider()

st.subheader("Comparison")

with st.expander('Functions'):
    col1, col2 = st.columns(2)
    with col1:
        st.help(plt.subplots)
    with col2:
        st.help(plt.axes)

st.code('''
fig, axes = plt.subplots(1, 3, figsize=(10,10))
axes[0].imshow(data)
axes[0].set_title('Original')

axes[1].imshow(resizeDim)
axes[1].set_title('Dimension Resize')

axes[2].imshow(resizeAxis)
axes[2].set_title('Axis Resize')
''', language='python')    

if st.button('Generate All', key=1):
    if 'opencv_image' not in locals():
        st.error('Please upload an image first')
    else:
        resizeDim = cv.resize(opencv_image.copy(), (100,100))
        resizeAxis = cv.resize(opencv_image.copy(), None, fx=0.5, fy=0.5)
        fig, axes = plt.subplots(1, 3, figsize=(10,10))
        axes[0].imshow(opencv_image.copy())
        axes[0].set_title('Original')
        axes[1].imshow(resizeDim)
        axes[1].set_title('Dimension Resize')
        axes[2].imshow(resizeAxis)
        axes[2].set_title('Axis Resize')
        st.pyplot(fig)

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
rotate = cv2.rotate(data.copy(), cv2.ROTATE_180)

fig, ax = plt.subplots()
ax.imshow(rotate)
ax.set_title('Rotate')
''', language='python')    

if st.button('Generate Rotate'):
    if 'opencv_image' not in locals():
        st.error('Please upload an image first')
    else:
        rotate = cv.rotate(opencv_image.copy(), cv.ROTATE_180)
        st.image(rotate)

st.divider()

st.subheader("Flip")

with st.expander('Functions'):
    st.help(cv.flip)

st.code('''
flip = cv2.flip(data.copy(), 90)

fig, ax = plt.subplots()
ax.imshow(flip)
ax.set_title('Flip')
''', language='python')    

if st.button('Generate Flip'):
    if 'opencv_image' not in locals():
        st.error('Please upload an image first')
    else:
        flip = cv.flip(opencv_image.copy(), 90)
        st.image(flip)

st.divider()

st.subheader("Region Of Interest")

with st.expander('Functions'):
    st.help(cv.warpAffine)

st.code('''
M = np.float32([[1,0,10],[0,1,10]])
roi = cv2.warpAffine(data.copy(), M, dsize=(100, 100))

fig, ax = plt.subplots()
ax.imshow(roi)
ax.set_title('Region Of Interest')
''', language='python')    

if st.button('Generate ROI'):
    if 'opencv_image' not in locals():
        st.error('Please upload an image first')
    else:
        M = np.float32([[1,0,10],[0,1,10]])
        roi = cv.warpAffine(opencv_image.copy(), M, dsize=(100, 100))
        st.image(roi)

st.divider()

st.subheader("Comparison")

with st.expander('Functions'):
    col1, col2 = st.columns(2)
    with col1:
        st.help(plt.subplots)
    with col2:
        st.help(plt.axes)

st.code('''
fig , axes = plt.subplots(1, 4, figsize=(10,10))
axes[0].imshow(data)
axes[0].set_title('Original')

axes[1].imshow(rotate)
axes[1].set_title('Rotate')

axes[2].imshow(flip)
axes[2].set_title('Flip')

axes[3].imshow(roi)
axes[3].set_title('ROI')
''', language='python')    

if st.button('Generate All', key=2):
    if 'opencv_image' not in locals():
        st.error('Please upload an image first')
    else:
        rotate = cv.rotate(opencv_image.copy(), cv.ROTATE_180)
        flip = cv.flip(opencv_image.copy(), 90)
        M = np.float32([[1,0,10],[0,1,10]])
        roi = cv.warpAffine(opencv_image.copy(), M, dsize=(100, 100))
        fig, axes = plt.subplots(1, 4, figsize=(10,10))
        axes[0].imshow(opencv_image.copy())
        axes[0].set_title('Original')
        axes[1].imshow(rotate)
        axes[1].set_title('Rotate')
        axes[2].imshow(flip)
        axes[2].set_title('Flip')
        axes[3].imshow(roi)
        axes[3].set_title('ROI')
        st.pyplot(fig)

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
blur = cv2.blur(data.copy(), ksize=(9,9))

fig, ax = plt.subplots()
ax.imshow(blur)
ax.set_title('Blur')
''', language='python')    

if st.button('Generate Blur'):
    if 'opencv_image' not in locals():
        st.error('Please upload an image first')
    else:
        blur = cv.blur(opencv_image.copy(), ksize=(9,9))
        st.image(blur)

st.divider()

st.subheader("Gaussian Blur")

with st.expander('Functions'):
    st.help(cv.GaussianBlur)

st.code('''
GaussianBlur = cv2.GaussianBlur(data.copy(), (9,9), 0)

fig, ax = plt.subplots()
ax.imshow(GaussianBlur)
ax.set_title('Gaussian Blur')
''', language='python')    

if st.button('Generate Gaussian Blur'):
    if 'opencv_image' not in locals():
        st.error('Please upload an image first')
    else:
        GaussianBlur = cv.GaussianBlur(opencv_image.copy(), (9,9), 0)
        st.image(GaussianBlur)

st.divider()

st.subheader("Median Blur")

with st.expander('Functions'):
    st.help(cv.medianBlur)

st.code('''
medianBlur = cv2.medianBlur(data.copy(), 9, 0)

fig, ax = plt.subplots()
ax.imshow(medianBlur)
ax.set_title('Median Blur')
''', language='python')    

if st.button('Generate Median Blur'):
    if 'opencv_image' not in locals():
        st.error('Please upload an image first')
    else:
        medianBlur = cv.medianBlur(opencv_image.copy(), 9)
        st.image(medianBlur)

st.divider()

st.subheader("Noise Reduction")

with st.expander('Functions'):
    st.help(cv.fastNlMeansDenoisingColored)

st.code('''
deNoise = cv2.fastNlMeansDenoisingColored(data.copy(), h=10)
fig, ax = plt.subplots()
ax.imshow(deNoise)
ax.set_title('Noise Reduction')
''', language='python')

if st.button('Generate Noise Reduction'):
    if 'opencv_image' not in locals():
        st.error('Please upload an image first')
    else:
        deNoise = cv.fastNlMeansDenoisingColored(opencv_image.copy(), h=10)
        st.image(deNoise)

st.divider()

st.subheader("Comparison")

with st.expander('Functions'):
    col1, col2 = st.columns(2)
    with col1:
        st.help(plt.subplots)
    with col2:
        st.help(plt.axes)

st.code('''
fig , axes = plt.subplots(2,3, figsize=(10,10))
axes[0,0].imshow(data)
axes[0,0].set_title('Original')
axes[0,1].imshow(blur)
axes[0,1].set_title('Blur')
axes[0,2].imshow(GaussianBlur)
axes[0,2].set_title('Gaussian Blur')
axes[1,0].imshow(medianBlur)
axes[1,0].set_title('Median Blur')
axes[1,1].imshow(deNoise)
axes[1,1].set_title('Noise Reduction')
axes[1,2].axis('off')
''', language='python')    

if st.button('Generate All', key=3):
    if 'opencv_image' not in locals():
        st.error('Please upload an image first')
    else:
        blur = cv.blur(opencv_image.copy(), ksize=(9,9))       
        GaussianBlur = cv.GaussianBlur(opencv_image.copy(), (9,9), 0)
        medianBlur = cv.medianBlur(opencv_image.copy(), 9)
        deNoise = cv.fastNlMeansDenoisingColored(opencv_image.copy(), h=10)
        fig, axes = plt.subplots(2,4, figsize=(10,10))
        axes[0,0].imshow(opencv_image.copy())
        axes[0,0].set_title('Original')
        axes[0,1].imshow(blur)
        axes[0,1].set_title('Blur')
        axes[0,2].imshow(GaussianBlur)
        axes[0,2].set_title('Gaussian Blur')
        axes[1,0].imshow(medianBlur)
        axes[1,0].set_title('Median Blur')
        axes[1,1].imshow(deNoise)
        axes[1,1].set_title('Noise Reduction')
        axes[1,2].axis('off')
        axes[1,3].axis('off')
        axes[0,3].axis('off')
        st.pyplot(fig)

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
canny = cv2.Canny(data.copy(), 100, 200)

fig, ax = plt.subplots()
ax.imshow(canny)
ax.set_title('Canny')
''', language='python')

if st.button('Generate Canny'):
    if 'opencv_image' not in locals():
        st.error('Please upload an image first')
    else:
        canny = cv.Canny(opencv_image.copy(), 100, 200)
        st.image(canny)

st.divider()

st.subheader("Sobel")

with st.expander('Functions'):
    st.help(cv.Sobel)

st.code('''
sobel = cv2.Sobel(data.copy(),ddepth= 5, dx=1, dy=1, ksize=3)

fig, ax = plt.subplots()
ax.imshow(sobel)
ax.set_title('Sobel')
''', language='python')

if st.button('Generate Sobel'):
    if 'opencv_image' not in locals():
        st.error('Please upload an image first')
    else:
        sobel = cv.Sobel(opencv_image.copy(), ddepth=5, dx=1,dy= 1, ksize=3)
        sobel = np.uint8(sobel * 255 / np.max(sobel))
        st.image(sobel)

st.divider()

st.subheader("Scharr")

with st.expander('Functions'):
    st.help(cv.Scharr)

st.code('''
scharr = cv2.Scharr(data.copy(),ddepth= 5, dx=1, dy=0)

fig, ax = plt.subplots()
ax.imshow(scharr)
ax.set_title('Scharr')
''', language='python')

if st.button('Generate Scharr'):
    if 'opencv_image' not in locals():
        st.error('Please upload an image first')
    else:
        scharr = cv.Scharr(opencv_image.copy(), ddepth= 5, dx=1, dy=0)
        scharr = np.uint8(scharr * 255 / np.max(scharr))
        st.image(scharr)

st.divider()

st.subheader("Laplacian")

with st.expander('Functions'):
    st.help(cv.Laplacian)

st.code('''
laplacian = cv2.Laplacian(data.copy(), ddepth=5, ksize=3)

fig, ax = plt.subplots()
ax.imshow(laplacian)
ax.set_title('Laplacian')
''', language='python')

if st.button('Generate Laplacian'):
    if 'opencv_image' not in locals():
        st.error('Please upload an image first')
    else:
        laplacian = cv.Laplacian(opencv_image.copy(), ddepth=5, ksize=3)
        laplacian = np.uint8(laplacian * 255 / np.max(laplacian))
        st.image(laplacian)

st.divider()

st.subheader("Erode")

with st.expander('Functions'):
    st.help(cv.erode)

st.code('''
kernel = np.ones((5,5),np.uint8)
erode = cv2.erode(data.copy(), kernel)

fig, ax = plt.subplots()
ax.imshow(erode)
ax.set_title('Erode')
''', language='python')

if st.button('Generate Erode'):
    if 'opencv_image' not in locals():
        st.error('Please upload an image first')
    else:
        kernel = np.ones((5,5),np.uint8)
        erode = cv.erode(opencv_image.copy(), kernel)
        st.image(erode)

st.divider()

st.subheader("Dilate")
with st.expander('Functions'):
    st.help(cv.dilate)

st.code('''
kernel = np.ones((5,5),np.uint8)
dilate = cv2.dilate(data.copy(), kernel)

fig, ax = plt.subplots()
ax.imshow(dilate)
ax.set_title('Dilate')
''', language='python')

if st.button('Generate Dilate'):
    if 'opencv_image' not in locals():
        st.error('Please upload an image first')
    else:
        kernel = np.ones((5,5),np.uint8)    
        dilate = cv.dilate(opencv_image.copy(), kernel)
        st.image(dilate)

st.divider()

st.subheader("Contour")

with st.expander('Functions'):
    col1, col2 = st.columns(2)
    with col1:
        st.help(cv.findContours)
    with col2:
        st.help(cv.drawContours)
    
st.code('''
contours, hierarchy = cv2.findContours(grey.copy(), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
contour = cv2.drawContours(data.copy(), contours, contourIdx=-1, color=(0,255,0), thickness=2)

fig, ax = plt.subplots()
ax.imshow(contour)
ax.set_title('Contours')
''', language='python')

if st.button('Generate Contour'):
    if 'opencv_image' not in locals():
        st.error('Please upload an image first')
    else:
        grey = cv.cvtColor(opencv_image.copy(), cv.COLOR_BGR2GRAY)
        contours, hierarchy = cv.findContours(grey, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
        contour = cv.drawContours(opencv_image.copy(), contours, contourIdx=-1, color=(0,255,0), thickness=2)
        st.image(contour)

st.divider()

st.subheader("Comparison")

with st.expander('Functions'):
    col1, col2 = st.columns(2)
    with col1:
        st.help(plt.subplots)
    with col2:
        st.help(plt.axes)

st.code('''
fig , axes = plt.subplots(3,3, figsize=(10,10))
axes[0,0].imshow(data)
axes[0,0].set_title('Original')

axes[0,1].imshow(canny)
axes[0,1].set_title('Canny')

axes[0,2].imshow(sobel)
axes[0,2].set_title('Sobel')

axes[1,0].imshow(scharr)
axes[1,0].set_title('Scharr')

axes[1,1].imshow(laplacian)
axes[1,1].set_title('Laplacian')

axes[1,2].imshow(erode)
axes[1,2].set_title('Erode')

axes[2,0].imshow(dilate)
axes[2,0].set_title('Dilate')

axes[2,1].imshow(contour)
axes[2,1].set_title('Contours')
        
axes[2,2].axis('off')
''', language='python')

if st.button('Generate All', key=4):
    if 'opencv_image' not in locals():
        st.error('Please upload an image first')
    else:
        canny = cv.Canny(opencv_image.copy(), 100, 200)
        sobel = cv.Sobel(opencv_image.copy(), ddepth=5, dx=1,dy= 1, ksize=3)
        sobel = np.uint8(sobel * 255 / np.max(sobel))
        scharr = cv.Scharr(opencv_image.copy(), ddepth= 5, dx=1, dy=0)
        scharr = np.uint8(scharr * 255 / np.max(scharr))
        laplacian = cv.Laplacian(opencv_image.copy(), ddepth=5, ksize=3)
        laplacian = np.uint8(laplacian * 255 / np.max(laplacian))
        kernel = np.ones((5,5),np.uint8)
        erode = cv.erode(opencv_image.copy(), kernel)
        dilate = cv.dilate(opencv_image.copy(), kernel)
        grey = cv.cvtColor(opencv_image.copy(), cv.COLOR_BGR2GRAY)
        contours, hierarchy = cv.findContours(grey, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
        contour = cv.drawContours(opencv_image.copy(), contours, contourIdx=-1, color=(0,255,0), thickness=2)
        fig, axes = plt.subplots(3,3, figsize=(10,10))
        axes[0,0].imshow(opencv_image.copy())
        axes[0,0].set_title('Original')
        axes[0,1].imshow(canny)
        axes[0,1].set_title('Canny')
        axes[0,2].imshow(sobel)
        axes[0,2].set_title('Sobel')
        axes[1,0].imshow(scharr)
        axes[1,0].set_title('Scharr')
        axes[1,1].imshow(laplacian)
        axes[1,1].set_title('Laplacian')
        axes[1,2].imshow(erode)
        axes[1,2].set_title('Erode')
        axes[2,0].imshow(dilate)
        axes[2,0].set_title('Dilate')
        axes[2,1].imshow(contour)
        axes[2,1].set_title('Contours')
        axes[2,2].axis('off')
        st.pyplot(fig)

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
line = cv2.line(data.copy(),pt1=(50, 50), pt2=(200, 150), color=(0, 255, 0), thickness=2)

fig, ax = plt.subplots()
ax.imshow(line)
ax.set_title('Line')
''', language='python')

if st.button('Generate Line'):
    if 'opencv_image' not in locals():
        st.error('Please upload an image first')
    else:
        line = cv.line(opencv_image.copy(), pt1=(50, 50), pt2=(200, 150), color=(0, 255, 0), thickness=2)
        st.image(line)

st.divider()

st.subheader("Rectangle")

with st.expander('Functions'):
    st.help(cv.rectangle)

st.code('''
rectangle = cv2.rectangle(data.copy(), pt1=(50, 50), pt2=(200, 150), color=(0, 255, 0), thickness=2)

fig, ax = plt.subplots()
ax.imshow(rectangle)
ax.set_title('Rectangle line')
''', language='python')

if st.button('Generate Rectangle'):
    if 'opencv_image' not in locals():
        st.error('Please upload an image first')
    else:
        rectangle = cv.rectangle(opencv_image.copy(), pt1=(50, 50), pt2=(200, 150), color=(0, 255, 0), thickness=2)
        st.image(rectangle)

st.divider()

st.subheader("Circle")
with st.expander('Functions'):
    st.help(cv.circle)

st.code('''
circle = cv2.circle(data.copy(), center=(100,150), radius=50, color=(0,255,0), thickness=2)

fig, ax = plt.subplots()
ax.imshow(circle)
ax.set_title('Circle line')
''', language='python')

if st.button('Generate Circle'):
    if 'opencv_image' not in locals():
        st.error('Please upload an image first')
    else:
        circle = cv.circle(opencv_image.copy(), center=(100,150), radius=50, color=(0,255,0), thickness=2)
        st.image(circle)

st.divider()

st.subheader("Text Drawing")

with st.expander('Functions'):
    st.help(cv.putText)

st.code('''
text = cv2.putText(data.copy(), text='Hello World', org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2)

fig, ax = plt.subplots()
ax.imshow(text)
ax.set_title('Text Drawing')
''', language='python')

if st.button('Generate Text'):
    if 'opencv_image' not in locals():
        st.error('Please upload an image first')
    else:
        text = cv.putText(opencv_image.copy(), text='Hello World', org=(50, 50), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2)
        st.image(text)

st.divider()

st.subheader("Comparison")

with st.expander('Functions'):
    col1, col2 = st.columns(2)
    with col1:
        st.help(plt.subplots)
    with col2:
        st.help(plt.axes)

st.code('''
fig, axes = plt.subplots(2,3, figsize=(10,10))

axes[0,0].imshow(data)
axes[0,0].set_title('Original')

axes[0,1].imshow(line)
axes[0,1].set_title('Line')

axes[0,2].imshow(circle)
axes[0,2].set_title('Circle')
        
axes[1,0].imshow(rectangle)
axes[1,0].set_title('Rectangle')
        
axes[1,1].imshow(text)
axes[1,1].set_title('Text')
axes[1,2].axis('off')
''', language='python')

if st.button('Generate All', key=5):
    if 'opencv_image' not in locals():
        st.error('Please upload an image first')
    else:
        line = cv.line(opencv_image.copy(), pt1=(50, 50), pt2=(200, 150), color=(0, 255, 0), thickness=2)
        rectangle = cv.rectangle(opencv_image.copy(), pt1=(50, 50), pt2=(200, 150), color=(0, 255, 0), thickness=2)
        circle = cv.circle(opencv_image.copy(), center=(100,150), radius=50, color=(0,255,0), thickness=2)
        text = cv.putText(opencv_image.copy(), text='Hello World', org=(50, 50), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2)
        fig, axes = plt.subplots(2,3, figsize=(10,10))
        axes[0,0].imshow(opencv_image.copy())
        axes[0,0].set_title('Original')
        axes[0,1].imshow(line)
        axes[0,1].set_title('Line')
        axes[0,2].imshow(circle)
        axes[0,2].set_title('Circle')
        axes[1,0].imshow(rectangle)
        axes[1,0].set_title('Rectangle')
        axes[1,1].imshow(text)
        axes[1,1].set_title('Text')
        axes[1,2].axis('off')
        st.pyplot(fig)