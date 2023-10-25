from fastai.vision.all import *
import gradio as gr
from fastai.vision.all import PILImage

learn = load_learner('model.pkl')
LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def get_img(data):
    t1 = data.reshape(28,28)/255
    t1 = np.stack([t1]*3,axis=0)
    img = Image(FloatTensor(t1))
    return img


def predict(img):
    print(f"type: {img}")
    testimg = PILImage.create(img)

    imgtens = get_img(img)
    print(f"type: {imgtens}")
    pred, idx, probs = learn.predict(imgtens)
    print(f"pred: {learn.predict(imgtens)}")

    #confidences = {LABELS[i]: v.item() for i, v in zip(pred, probs)}
    return dict(zip(LABELS, map(float,probs)))


label = gr.outputs.Label()

sp = gr.Sketchpad(shape=(28, 28),  image_mode="L")

gr.Interface(fn=predict,
             inputs=sp,
             outputs="label",
        ).queue().launch(share=True)