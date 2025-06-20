# Vision.edu - Image and Video Processing Web Application
# 🔗 Link - https://visionedu.streamlit.app

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.0+-red.svg)
![OpenCV](https://img.shields.io/badge/opencv-v4.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**Vision.edu** is a comprehensive web-based image and video processing application built with Streamlit and OpenCV. It provides an intuitive interface for performing various computer vision operations including filtering, edge detection, color space conversion, geometric transformations, and drawing operations.

## 🌟 Features

### 🎨 Color Space Conversion
- **Gray Scale**: Convert images/videos to grayscale
- **HSV Scale**: Transform images to HSV color space for better color analysis

### 📏 Resizing Operations
- **Axis Resize**: Resize images along specific axes
- **Dimension Resize**: Resize images by custom dimensions

### 🔄 Geometric Transformations
- **Flip**: Horizontal and vertical image flipping
- **Rotate**: Rotate images by custom angles
- **ROI (Region of Interest)**: Extract specific regions from images

### 🔍 Filtering & Noise Reduction
- **Blur**: Basic blur filter application
- **Gaussian Blur**: Advanced Gaussian blur with customizable kernels
- **Median Blur**: Median filtering for noise reduction
- **Noise Reduction**: Advanced noise reduction algorithms

### 🔲 Edge Detection
- **Canny Edge**: Industry-standard Canny edge detection
- **Laplacian Edge**: Laplacian operator for edge detection
- **Scharr Edge**: Scharr operator for precise edge detection
- **Sobel Edge**: Sobel operator for gradient-based edge detection
- **Erode Edge**: Morphological erosion operations
- **Dilate Edge**: Morphological dilation operations
- **Contour Edge**: Contour detection and visualization

### ✏️ Drawing Operations
- **Line Draw**: Draw custom lines on images
- **Rectangle Draw**: Add rectangular shapes
- **Circle Draw**: Insert circular shapes
- **Text Draw**: Add custom text overlays

### 🎯 Master Applications
- **Image Processing Master**: Complete suite of image processing tools
- **Video Processing Master**: Comprehensive video processing capabilities

## 🚀 Quick Start

### Prerequisites

Ensure you have Python 3.8 or higher installed on your system.

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/rounakdey2003/vision-edu.git
   cd vision-edu
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run VisionEdu.py
   ```

4. **Access the application**
   Open your web browser and navigate to `http://localhost:8501`

## 📦 Dependencies

The application requires the following Python packages:

```
streamlit           # Web application framework
opencv-python       # Computer vision library
numpy              # Numerical computing
matplotlib         # Plotting and visualization
Pillow             # Image processing library
```

## 🏗️ Project Structure

```
ImageManipulator/
├── VisionEdu.py              # Main application entry point
├── requirements.txt          # Project dependencies
├── cat.jpg                   # Sample image file
├── cat.mp4                   # Sample video file
├── README.md                 # Project documentation
└── pages/                    # Individual processing modules
    ├── Axis Resize.py        # Axis-based resizing
    ├── Blur.py               # Basic blur operations
    ├── Canny Edge.py         # Canny edge detection
    ├── Circle Draw.py        # Circle drawing tools
    ├── Contour Edge.py       # Contour detection
    ├── Dilate Edge.py        # Dilation operations
    ├── Dimension Resize.py   # Dimension-based resizing
    ├── Erode Edge.py         # Erosion operations
    ├── Flip Representation.py # Image flipping
    ├── Gaussian Blur.py      # Gaussian blur filtering
    ├── Gray Scale.py         # Grayscale conversion
    ├── Hsv Scale.py          # HSV color space conversion
    ├── Image Processing.py   # Master image processing
    ├── Laplacian Edge.py     # Laplacian edge detection
    ├── Line Draw.py          # Line drawing tools
    ├── Median Blur.py        # Median blur filtering
    ├── Noise Reduction.py    # Noise reduction algorithms
    ├── Rectangle Draw.py     # Rectangle drawing tools
    ├── Roi Representation.py # ROI extraction
    ├── Rotate Representation.py # Image rotation
    ├── Scharr Edge.py        # Scharr edge detection
    ├── Sobel Edge.py         # Sobel edge detection
    ├── Text Draw.py          # Text overlay tools
    ├── Video Processing.py   # Master video processing
    └── __pycache__/          # Compiled Python files
```

## 🎯 Usage Guide

### Getting Started

1. **Launch the application** by running `streamlit run VisionEdu.py`
2. **Upload your media** using the file uploader in any processing module
3. **Adjust parameters** using the interactive sliders and controls
4. **View results** in real-time as you modify settings
5. **Download processed files** using the download buttons

### Navigation

- **Search Functionality**: Use the search bar to quickly find specific processing tools
- **Category Browsing**: Explore tools organized by categories (Color, Resize, Edge Detection, etc.)
- **Master Apps**: Access comprehensive processing suites for batch operations

### Supported File Formats

**Images**: JPG, JPEG, PNG
**Videos**: MP4, AVI, MOV

## 🔧 Key Features in Detail

### Real-time Processing
All operations are performed in real-time, allowing you to see immediate results as you adjust parameters.

### Interactive Controls
- Sliders for numerical parameters (brightness, contrast, kernel sizes)
- Checkboxes for boolean options
- Dropdown menus for selection options
- File uploaders for media input

### Parameter Customization
Each tool provides relevant parameters for fine-tuning:
- **Blur operations**: Kernel size adjustment
- **Edge detection**: Threshold values
- **Drawing tools**: Color, thickness, and position controls
- **Geometric transforms**: Angle, scale, and axis parameters

### Download Capabilities
Processed images and videos can be downloaded directly from the interface.

## 🛠️ Development

### Adding New Features

1. Create a new Python file in the `pages/` directory
2. Follow the existing module structure:
   ```python
   import streamlit as st
   import cv2 as cv
   import numpy as np
   
   st.set_page_config(
       page_title="Your Feature",
       page_icon="🧊",
       layout="centered",
       initial_sidebar_state="expanded",
   )
   
   # Your processing logic here
   ```
3. Add the new feature to the main navigation in `VisionEdu.py`

### Code Style

- Follow PEP 8 Python style guidelines
- Use descriptive variable names
- Include comments for complex operations
- Maintain consistent error handling

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenCV**: For providing comprehensive computer vision capabilities
- **Streamlit**: For the excellent web application framework
- **NumPy**: For efficient numerical operations
- **Matplotlib**: For visualization capabilities
- **Pillow**: For additional image processing support

## 📧 Contact

**Developer**: Rounak Dey
**GitHub**: [@rounakdey2003](https://github.com/rounakdey2003)

## 🔮 Future Enhancements

- [ ] Machine Learning integration for advanced image analysis
- [ ] Batch processing capabilities
- [ ] Custom filter creation tools
- [ ] Integration with cloud storage services
- [ ] Mobile-responsive design improvements
- [ ] Video streaming capabilities
- [ ] Advanced annotation tools
- [ ] Export to multiple formats
- [ ] User account system with saved projects
- [ ] API endpoints for programmatic access

## 📊 Performance Notes

- **Image Processing**: Optimized for images up to 4K resolution
- **Video Processing**: Recommended for videos under 100MB for optimal performance
- **Real-time Preview**: Efficient algorithms ensure smooth real-time processing
- **Memory Management**: Automatic cleanup prevents memory leaks during extended use

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed via `pip install -r requirements.txt`
2. **File Upload Issues**: Check file format compatibility (JPG, PNG, MP4, AVI, MOV)
3. **Performance Issues**: For large files, consider resizing before processing
4. **Browser Compatibility**: Use modern browsers (Chrome, Firefox, Safari, Edge)

### Getting Help

If you encounter issues:
1. Check the [Issues](https://github.com/rounakdey2003/vision-edu/issues) section
2. Create a new issue with detailed description
3. Include error messages and system information

---

⭐ **Star this repository if you find it helpful!**
