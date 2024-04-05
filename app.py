import cv2
import torch

import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from models import *
import plotly.graph_objects as go

st.title("Number recognition by sai nivedh")

mode=st.checkbox('Draw or transform',True)
col1,col2 = st.columns([1,1])
with col1:
    for i in range(5):
        st.write(' ')
canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=256,
    height=256,
    drawing_mode="freedraw" if mode else "transform",
    key='canvas'
)

pred_btn = st.button('Predict')

if canvas_result.image_data is not None:
    img=cv2.resize(canvas_result.image_data.astype('uint8'),(28,28))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=img[:,:,np.newaxis]/255.0 #(28,28) -> (1,28,28)

model=DigitModel()
model.load_state_dict(torch.load('best-wt.pt'))


with torch.no_grad():
    img=torch.Tensor(img).permute(2,0,1)
    logits=model(img.unsqueeze(0))
    sm=torch.nn.Softmax(dim=1)
    probs=sm(logits)[0]

    fig = go.Figure(
        data=[
            go.Bar(
                x=np.arange(0,10),
                y=(probs*100).numpy()
            )
        ]
    )
    fig.update_layout(
        width=500,
        height=500
    )
    st.plotly_chart(fig)