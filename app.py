from fastai.vision.all import *
import gradio as gr
from fastai.vision.all import PILImage

learn = load_learner('model.pkl')
categories = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

def predict(img):
    pred, idx, probs = learn.predict(PILImage.create(img))
    return dict(zip(categories, map(float,probs)))

label = gr.outputs.Label()
sp = gr.Sketchpad(shape=(28, 28),  image_mode="L")

intf = gr.Interface(fn=predict, inputs=sp, outputs=label)
intf.launch(inline=False)