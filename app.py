import gradio as gr
import joblib
import numpy as np
import pandas as pd

# Load the model and unique brand values
model = joblib.load('model.joblib')
unique_values = joblib.load('unique_values.joblib')
brand_values = unique_values['Brand']

# Define the prediction function
def predict(brand, screen_size, resolution_width, resolution_height):
    # Convert inputs to appropriate types
    screen_size = float(screen_size)
    resolution_width = int(resolution_width)
    resolution_height = int(resolution_height)
    
    # Prepare the input array for prediction
    input_data = pd.DataFrame({
        'Brand': [brand],
        'Screen Size': [screen_size],
        'Resolution (Width)': [resolution_width],
        'Resolution (Height)': [resolution_height]
    })
    
    # Perform the prediction
    prediction = model.predict(input_data)
    
    return prediction[0]

# Create the Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(choices=list(brand_values), label="Brand"),
        gr.Textbox(label="Screen Size"),
        gr.Textbox(label="Resolution (Width)"),
        gr.Textbox(label="Resolution (Height)")
    ],
    outputs="text",
    title="Monitor Predictor",
    description="Enter the brand, screen size, and resolution to predict the target value."
)

# Launch the app
interface.launch()
