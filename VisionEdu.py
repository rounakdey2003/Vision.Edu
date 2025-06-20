import time

import streamlit as st


st.set_page_config(
    page_title="Image Processing",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.spinner('Loading...'):
    st.toast('Checking Environment')

    st.title('**Vision**:grey[.edu]')

    subtitle = '**Manipulate Images and Videos in real time.**'


    def stream_data():
        for word in subtitle.split(" "):
            yield word + " "
            time.sleep(0.1)


    if ("Stream data"):
        st.write_stream(stream_data)

    st.divider()

    searchCol1, searchCol2, searchCol3 = st.columns([1, 3, 1])

    with searchCol1:
        pass
    with searchCol2:
        search_query = st.text_input(label='', placeholder='Search for a page',
                                     help=':grey[**Keywords**] = Scale, Resize, Flip, Blur, Edge, Draw')

        pages = [
        
            {"label": ":green-background[:green[Axis Resize]]", "page": "pages/Axis Resize.py",
             "help": "Resize image along axes"},
            {"label": ":green-background[:green[Blur]]", "page": "pages/Blur.py",
             "help": "Apply blur filter to image"},
            {"label": ":green-background[:green[Canny Edge]]", "page": "pages/Canny Edge.py",
             "help": "Apply Canny edge detection to image"},
            {"label": ":green-background[:green[Circle Draw]]", "page": "pages/Circle Draw.py",
             "help": "Draw circles on image"},
            {"label": ":green-background[:green[Contour Edge]]", "page": "pages/Contour Edge.py",
             "help": "Detect and draw contours on image"},
            {"label": ":green-background[:green[Dilate Edge]]", "page": "pages/Dilate Edge.py",
             "help": "Apply dilation to image"},
            {"label": ":green-background[:green[Dimension Resize]]", "page": "pages/Dimension Resize.py",
             "help": "Resize image by dimensions"},
            {"label": ":green-background[:green[Erode Edge]]", "page": "pages/Erode Edge.py",
             "help": "Apply erosion to image"},
            {"label": ":green-background[:green[Flip]]", "page": "pages/Flip Representation.py",
             "help": "Flip image"},
            {"label": ":green-background[:green[Gaussian Blur]]", "page": "pages/Gaussian Blur.py",
             "help": "Apply Gaussian blur filter to image"},
            {"label": ":green-background[:green[Gray Scale]]", "page": "pages/Gray Scale.py",
             "help": "Convert image to grayscale"},
            {"label": ":green-background[:green[HSV Scale]]", "page": "pages/Hsv Scale.py",
             "help": "Convert image to HSV color space"},
            {"label": ":green-background[:green[Laplacian Edge]]", "page": "pages/Laplacian Edge.py",
             "help": "Apply Laplacian edge detection to image"},
            {"label": ":green-background[:green[Line Draw]]", "page": "pages/Line Draw.py",
             "help": "Draw lines on image"},
            {"label": ":green-background[:green[Median Blur]]", "page": "pages/Median Blur.py",
             "help": "Apply median blur filter to image"},
            {"label": ":green-background[:green[Rectangle Draw]]", "page": "pages/Rectangle Draw.py",
             "help": "Draw rectangles on image"},
            {"label": ":green-background[:green[ROI]]", "page": "pages/Roi Representation.py",
             "help": "Extract Region of Interest from image"},
            {"label": ":green-background[:green[Rotate]]", "page": "pages/Rotate Representation.py",
             "help": "Rotate image"},
            {"label": ":green-background[:green[Scharr Edge]]", "page": "pages/Scharr Edge.py",
             "help": "Apply Scharr edge detection to image"},
            {"label": ":green-background[:green[Sobel Edge]]", "page": "pages/Sobel Edge.py",
             "help": "Apply Sobel edge detection to image"},
            {"label": ":green-background[:green[Text Draw]]", "page": "pages/Text Draw.py",
             "help": "Draw text on image"},
            {"label": ":green-background[:green[Noise Reduction]]", "page": "pages/Noise Reduction.py",
             "help": "Reduce noise in image"},
            {"label": ":green-background[:green[Image Processing:grey[ Master]]]", "page": "pages/Image Processing.py",
             "help": "A collection of all the image processing methods"},
            {"label": ":green-background[:green[Video Processing:grey[ Master]]]", "page": "pages/Video Processing.py",
             "help": "A collection of all the video processing methods"},
        ]

        if search_query:
            filtered_pages = [page for page in pages if search_query.lower() in page["label"].lower()]

            for page in filtered_pages:
                st.page_link(page=page["page"], label=page["label"], help=page["help"], use_container_width=False)

    with searchCol3:
        pass

    st.write('##')

    with st.container(height=100):
        col1, col2, col3 = st.columns(3)
        with col1:
            pass
        with col2:
            st.header("Vision Processing")
        with col3:
            pass

    bodyCol1, bodyCol2 = st.columns([1, 1])

    with bodyCol1:

        st.subheader(":grey[Colour Scalling]")

        with st.expander("Converting image or video from one colour space to another"):
        
            if st.button(":green[Gray Scale]"):
                st.switch_page("pages/Gray Scale.py")


            if st.button(":green[HSV Scale]"):
                st.switch_page("pages/Hsv Scale.py")

        st.divider()

        st.subheader(":grey[Resizing]")

        with st.expander("Changing the size of image or video"):

            if st.button(":green[Axis Resize]"):
                st.switch_page("pages/Axis Resize.py")

            if st.button(":green[Dimension Resize]"):
                st.switch_page("pages/Dimension Resize.py")

        st.divider()

        st.subheader(":grey[Representation]")

        with st.expander("Changing the orientation of image or video"):

            if st.button(":green[Flip]"):
                st.switch_page("pages/Flip Representation.py")

            if st.button(":green[Rotate]"):
                st.switch_page("pages/Rotate Representation.py")

            if st.button(":green[ROI]"):
                st.switch_page("pages/Roi Representation.py")

        st.divider()

        
    with bodyCol2:

        st.subheader(":grey[Filtering]")

        with st.expander("Modifying the appearance of image or video"):

            if st.button(":green[Blur]"):
                st.switch_page("pages/Blur.py")

            if st.button(":green[Gaussian Blur]"):
                st.switch_page("pages/Gaussian Blur.py")

            if st.button(":green[Median Blur]"):
                st.switch_page("pages/Median Blur.py")

            if st.button(":green[Noise Reduction]"):
                st.switch_page("pages/Noise Reduction.py")

        st.divider()

        st.subheader(":grey[Edge Detection]")

        with st.expander("Detecting edges in image or video"):

            if st.button(":green[Canny Edge]"):
                st.switch_page("pages/Canny Edge.py")

            if st.button(":green[Laplacian Edge]"):
                st.switch_page("pages/Laplacian Edge.py")

            if st.button(":green[Scharr Edge]"):
                st.switch_page("pages/Scharr Edge.py")

            if st.button(":green[Sobel Edge]"):
                st.switch_page("pages/Sobel Edge.py")

            if st.button(":green[Erode Edge]"):
                st.switch_page("pages/Erode Edge.py")

            if st.button(":green[Dilate Edge]"):
                st.switch_page("pages/Dilate Edge.py")

            if st.button(":green[Contour Edge]"):
                st.switch_page("pages/Contour Edge.py")

        st.divider()

        st.subheader(":grey[Drawing]")

        with st.expander("Adding shapes to image or video"):

            if st.button(":green[Line Draw]"):
                st.switch_page("pages/Line Draw.py")


            if st.button(":green[Rectangle Draw]"):
                st.switch_page("pages/Rectangle Draw.py")

            if st.button(":green[Circle Draw]"):
                st.switch_page("pages/Circle Draw.py")

            if st.button(":green[Text Draw]"):
                st.switch_page("pages/Text Draw.py")

        st.divider()

    st.divider()

    st.subheader(":grey[Master App]")

    with st.expander("A collection of all the image and video processing methods"):
        if st.button(":green[Image Processing:grey[ Master]]"):
            st.switch_page("pages/Image Processing.py")

        if st.button(":green[Video Processing:grey[ Master]]"):
            st.switch_page("pages/Video Processing.py")

    st.divider()


st.page_link(page="https://github.com/rounakdey2003/Vision.Edu", label=":blue-background[:blue[Github]]",
                     help='Teleport to Github',
                     use_container_width=False)

st.toast(':green[Ready!]')
