!pip install gradio

import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
import gradio as gr


classes_dict = {
0:'Class_1_-_Tinea_unguium',
1:'Class_2_-_Paronychia',
2:'AppClass_3_-_Ingrown_Toenail',
3:'Class_4_-_Nail_Psoriasis',
4:'Class_5_-_Erythrasma',
5:'Class_6_-_Toenail_Melanoma',
6:'Class_7_-_Onychodystrophy',
7:'Class_8_-_Subungual_Hematoma',
8:'Class_9_-_Pincer_Nail',
9:'Class_10_-_Nail_Dystrophy',
}


# Remove underscores from the dictionary values
classes_dict_cleaned = {key: value.replace('_', ' ') for key, value in classes_dict.items()}

# Now, classes_dict_cleaned contains the cleaned class names
# print(classes_dict_cleaned)


# Function to make predictions

def prediction(path, model, classes_dict):
    # Load and preprocess the image
    img = load_img(path, target_size=(256, 256))
    img_arr = img_to_array(img)
    processed_img_arr = preprocess_input(img_arr)

    # Expand image dimensions
    img_exp_dim = np.expand_dims(processed_img_arr, axis=0)

    # Make predictions using the model
    pred = np.argmax(model.predict(img_exp_dim))

    # Get the predicted class label
    predicted_class = classes_dict[pred]

    # Make predictions using the model
    prediction_probabilities = model.predict(img_exp_dim)
    pred_class_index = np.argmax(prediction_probabilities)
    predicted_probability = prediction_probabilities[0][pred_class_index]

    # Plot the input image
    plt.imshow(img)
    plt.axis('off')  # Remove axis
    plt.title(f"Predicted Class: {predicted_class}")
    plt.show()

    return img, predicted_class


# Function called by the UI

def predict(image):
    model = load_model("../crop_health_monitoring_model.h5") #the path to where the model is saved.

    # path = "C:\\Users\\OMOLP094\\Desktop\\Research Projects\\Soham Jariwala - Crop Health Monitoring Project\\Crop Health Monitoring - Final Implementation\\test\\CornCommonRust1.JPG"


    path = image.name

    # Call the prediction function
    img, predicted_class = prediction(path, model, classes_dict_cleaned)

    return img, predicted_class


# User Interface

def main(): 
    io = gr.Interface(
        fn=predict,
        inputs=gr.File(label="Upload the image of the toenail", file_types = ["image"]),
        outputs = [gr.Image(label = "Uploaded Image", width = 400, height = 400), 
                   gr.Textbox(label="toenail Dieseases Overview")],
        allow_flagging="manual",
        flagging_options=["Save"],
        title="CNN based Common Toenail Diseases Classification",
        description="Classification of 10 Diseases Using Image Classification",
        theme = gr.themes.Soft()
    )

    io.launch(share=True)

if __name__ == "__main__":
    main()


