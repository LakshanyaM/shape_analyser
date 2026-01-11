import streamlit as st
import cv2
import numpy as np
from PIL import Image
import math

st.set_page_config(page_title="Shape & Contour Analyzer", layout="centered")
st.title("🔷 Shape & Contour Analyzer")

st.write("""
This application detects geometric shapes from an uploaded image using
**contour detection and feature extraction**.
It identifies shapes, counts objects, and computes **area and perimeter**.
""")


uploaded_file = st.file_uploader(
    "Upload an image (PNG / JPG / JPEG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    image_np = np.array(image)


    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(
        blurred, 200, 255, cv2.THRESH_BINARY_INV
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    output = image_np.copy()
    object_count = 0

    st.subheader("Detected Shapes")

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 500:
            continue

        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
        vertices = len(approx)

        circularity = (4 * math.pi * area) / (perimeter * perimeter)

        if circularity > 0.85:
            shape = "Circle"

        elif vertices == 3:
            shape = "Triangle"

        elif vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 0.95 <= aspect_ratio <= 1.05:
                shape = "Square"
            else:
                shape = "Rectangle"

        elif vertices == 5:
            shape = "Pentagon"

        elif vertices == 6:
            shape = "Hexagon"

        elif vertices == 7:
            shape = "Heptagon"

        elif vertices == 8:
            shape = "Octagon"

        else:
            shape = "Polygon"

        object_count += 1

        cv2.drawContours(output, [approx], -1, (0, 255, 0), 2)

        x, y = approx[0][0]
        cv2.putText(
            output,
            shape,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

        st.write(
            f"Shape: {shape} | Area: {int(area)} | Perimeter: {int(perimeter)}"
        )

    st.image(output, caption="Detected Shapes Output")
    st.success(f"Total Objects Detected: {object_count}")
