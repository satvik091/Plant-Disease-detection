
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

class PlantDiseaseClassifier:
    def __init__(self):
        # Mock model loading (replace with actual model path)
        self.model = self.load_model()
        self.class_names = [
            'Tomato Bacterial Spot', 
            'Tomato Early Blight', 
            'Tomato Healthy', 
            'Tomato Late Blight', 
            'Tomato Leaf Mold'
        ]

    def load_model(self):
        # Placeholder for model loading
        # In real implementation, use actual model loading
        return None

    def preprocess_image(self, image):
        # Resize and normalize image
        img = image.resize((256, 256))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict_disease(self, image):
        # Simulated prediction 
        processed_img = self.preprocess_image(image)
        
        # Mock prediction probabilities
        mock_probs = np.random.dirichlet(np.ones(len(self.class_names)), size=1)[0]
        predicted_class_index = np.argmax(mock_probs)
        
        return {
            'disease': self.class_names[predicted_class_index],
            'confidence': mock_probs[predicted_class_index],
            'probabilities': dict(zip(self.class_names, mock_probs))
        }

    def display_results(self, prediction):
        # Create visualization of results
        plt.figure(figsize=(10, 6))
        plt.bar(prediction['probabilities'].keys(), 
                prediction['probabilities'].values())
        plt.title('Disease Probability Distribution')
        plt.xlabel('Disease Classes')
        plt.ylabel('Probability')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return plt.gcf()

def main():
    st.set_page_config(
        page_title="Plant Disease Classifier", 
        page_icon="🌱"
    )

    # Initialize classifier
    classifier = PlantDiseaseClassifier()

    # App title and description
    st.title("Plant Disease Classification 🌿")
    st.write("Upload a plant leaf image to detect potential diseases")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a plant leaf image", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Prediction
        with st.spinner('Analyzing image...'):
            prediction = classifier.predict_disease(image)

        # Display results
        st.subheader("Classification Results")
        st.write(f"**Detected Disease:** {prediction['disease']}")
        st.write(f"**Confidence:** {prediction['confidence']:.2%}")

        # Probability distribution chart
        st.pyplot(classifier.display_results(prediction))

        # Treatment recommendations
        st.subheader("Recommended Actions")
        if prediction['disease'] == 'Tomato Healthy':
            st.success("Plant looks healthy! Continue regular care.")
        else:
            st.warning(f"Potential {prediction['disease']} detected. Consult agricultural expert.")

    # About section
    st.sidebar.title("About")
    st.sidebar.info(
        "This app uses machine learning to classify plant diseases "
        "from leaf images. Currently supports tomato plant diseases."
    )

if __name__ == "__main__":
    main()
