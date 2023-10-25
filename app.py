from fastai.vision.all import *
import gradio as gr

def is_cat(x): return x[0].isupper()

learn = load_learner('model.pkl')

categories = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

def classify_image(img):
    pred,idx,probs = learn.predict(img)
    print(f"Prediction: {pred}; Probability: {probs[0]:.4f}")
    return dict(zip(categories, map(float,probs)))

image = gr.components.Image(shape=(28, 28))
label = gr.components.Label()
examples = ['zero.png', 'one.png']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)