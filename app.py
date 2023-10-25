from fastai.vision.all import *
import gradio as gr

def is_cat(x): return x[0].isupper()

learn = load_learner('model.pkl')

categories = ('0', '1')

def classify_image(img):
    pred,idx,probs = learn.predict(img)
    print(f"Prediction: {pred}; Probability: {probs[idx]:.04f}")
    return dict(zip(categories, map(float,probs[idx]:.04f)))

image = gr.components.Image(shape=(192, 192))
label = gr.components.Label()
examples = ['zero.png', 'one.png']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)