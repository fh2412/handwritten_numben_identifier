from fastai.vision.all import *
import gradio as gr
from fastai.vision.all import PILImage

learn = load_learner('model.pkl')
LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def predict(img):
    print(f"type: {img}")
    print(f"type: {img[1:-1, 1:-1]}")
    testimg = PILImage.create(img)
    print(f"type: {testimg}")
    print(testimg.shape)
    #pred, idx, probs = learn.predict(img[0])
    pred, idx, probs = learn.predict(testimg)
    print(f"Prediction: {pred}; Probability: {probs[idx]:.04f}")

    #confidences = {LABELS[i]: v.item() for i, v in zip(pred, probs)}
    return dict(zip(LABELS, map(float,probs)))


label = gr.outputs.Label()

sp = gr.Sketchpad(shape=(28, 28),  image_mode="L")

gr.Interface(fn=predict,
             inputs=sp,
             outputs="label",
        ).queue().launch(share=True)