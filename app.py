from fastai.vision.all import *
import gradio as gr

learn = load_learner('model.pkl')

def predict(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))

gr.Interface(fn=predict,
             inputs="sketchpad",
             outputs="label",
             live=True).launch()