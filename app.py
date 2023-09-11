import cv2
import numpy as np
from tensorflow import keras
# streamlit
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# st.set_option('deprecation.showfileUploaderEncoding', False)

# ตั้งค่าขนาดหน้าจอ
st.set_page_config(page_title="My Streamlit App", page_icon=":guardsman:", layout="wide")
st.markdown(
    f"""
    <style>
        .reportview-container .main .block-container {{
            max-width: 100px;
            padding-top: 1rem;
            padding-right: 1rem;
            padding-left: 1rem;
            padding-bottom: 1rem;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Intelligent Calculation Application using Multiple Handwritten Digit Recognition.")
st.write("Upload an image of a handwritten mathematical expression.")
# Define the canvas
canvas = st_canvas(
    fill_color="#ffffff",
    stroke_width=5,
    stroke_color="#000000",
    background_color="#ffffff",
    height=180,
    width=1500,
    drawing_mode="freedraw",
    key="canvas"
)

# When the user clicks on "Predict"
if st.button("Predict"):
    model = keras.models.load_model("model_final")
    if canvas.image_data is not None:
        # Process the image
        img = cv2.cvtColor(canvas.image_data.astype("uint8"), cv2.COLOR_BGR2GRAY)
        img = cv2.bitwise_not(img)
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        ctrs, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
        w = int(28)
        h = int(28)
        train_data = []
        rects = []

        for c in cnt:
            x, y, w, h = cv2.boundingRect(c)
            rect = [x, y, w, h]
            rects.append(rect)

        bool_rect = []

        for r in rects:
            l = []
            for rec in rects:
                flag = 0
                if rec != r:
                    if r[0] < (rec[0] + rec[2] + 10) and rec[0] < (r[0] + r[2] + 10) and r[1] < (rec[1] + rec[3] + 10) and rec[1] < (r[1] + r[3] + 10):
                        flag = 1
                    l.append(flag)
                if rec == r:
                    l.append(0)
            bool_rect.append(l)

        dump_rect = []

        for i in range(0, len(cnt)):
            for j in range(0, len(cnt)):
                if bool_rect[i][j] == 1:
                    area1 = rects[i][2] * rects[i][3]
                    area2 = rects[j][2] * rects[j][3]
                    if (area1 == min(area1, area2)):
                        dump_rect.append(rects[i])

        final_rect = [i for i in rects if i not in dump_rect]

        for r in final_rect:
            x = r[0]
            y = r[1]
            w = r[2]
            h = r[3]
            im_crop = thresh[y:y + h + 10, x:x + w + 10]
            im_resize = cv2.resize(im_crop, (28, 28))
            im_resize = np.reshape(im_resize, (28, 28, 1))
            train_data.append(im_resize)

        equation = ''

        for i in range(len(train_data)):
            train_data[i] = np.array(train_data[i])
            train_data[i] = train_data[i].reshape(1, 28, 28, 1)
            result = np.argmax(model.predict(train_data[i]), axis=-1)

            for j in range(10):
                if result[0] == j:
                    equation = equation + str(j)

            if result[0] == 10:
                equation = equation + "+"
            if result[0] == 11:
                equation = equation + "-"
            if result[0] == 12:
                equation = equation + "*"
            if result[0] == 13:
                equation = equation + "*"
            if result[0] == 14:
                equation = equation + "/"
            if result[0] == 15:
                equation = equation + "/"
            if result[0] == 16:
                equation = equation + "."
                    
            print("Your Equation :", equation)

        # st.title('Handwritten OCR')
        st.write('<span style="font-size:28px">The evaluation of the image gives: </span>', unsafe_allow_html=True)
        st.write('<span style="font-size:28px">{}</span>'.format(equation), unsafe_allow_html=True)
        f = eval(equation)
        print('Result: ', '%.2f' %f)
        st.write('<span style="font-size:28px">Result:</span>', '<span style="font-size:28px">%.2f</span>' %f, unsafe_allow_html=True)