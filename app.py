from transformers import pipeline
import gradio as gr

# "Artifacts"
classifier = pipeline("image-classification", model="./my_final_image_model")

def predict(image):
    y_pred = classifier(image)
    y_pred = {y["label"]: y["score"] for y in y_pred}
    return y_pred

# https://www.gradio.app/guides
with gr.Blocks() as demo:
    image = gr.Image(type="pil")
    predict_btn = gr.Button("Predict", variant="primary")
    output = gr.Label(label="Output")

    inputs = [image]
    outputs = [output]
    
    predict_btn.click(predict, inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    demo.launch() # Local machine only
    # demo.launch(server_name="0.0.0.0") # LAN access to local machine
    # demo.launch(share=True) # Public access to local machine
