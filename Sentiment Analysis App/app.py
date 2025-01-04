import streamlit as st
from TextHelper import prepreoccessing  # Import the preprocessing function directly
import pickle
import nltk
import os
import streamlit.components.v1 as components  # Import to use HTML/CSS/JS

# Load model and vectorizer
model = pickle.load(open("Models/model.pkl", 'rb'))
vectorize = pickle.load(open("Models/vectorize.pkl", 'rb'))

# Specify the path to your custom nltk_data folder
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
# Ensure NLTK is looking in your custom data folder
nltk.data.path.append(nltk_data_path)

# Download the punkt resource to your custom folder (only once, no need to check)
nltk.download('punkt', download_dir=nltk_data_path)

# Now you can use the downloaded resources
st.title("Sentiment Analysis Using ML")

# Input and prediction
text = st.text_input('Please Enter your review')
state = st.button('Predict')

if state:
    token = prepreoccessing(text)  # Call the preprocessing function
    vectorize_data = vectorize.transform([token])
    prediction = model.predict(vectorize_data)
    
    # Create a colored rectangle around the sentiment result
    if prediction == [0]:
        sentiment = "The sentiment is: Negative"
        color = "#ff6347"  # Red for negative sentiment
        emotional_effect = "Sad"
        # Load the trees animation for negative sentiment
        with open("animations.html", "r") as f:
            animations = f.read()
        components.html(animations, height=800)
    elif prediction == [1]:
        sentiment = "The sentiment is: Positive"
        color = "#00bcd4"  # Blue for positive sentiment
        emotional_effect = "Happiest"
        # Load the balloons and stars animation for positive sentiment
        with open("animations.html", "r") as f:
            animations = f.read()
        components.html(animations, height=800)

    # Display the result in a colored box
    result_html = f"""
    <div style="background-color:{color}; padding: 20px; border-radius: 10px; text-align: center;">
        <h3>{sentiment}</h3>
        <p>Emotional effect: {emotional_effect}</p>
    </div>
    """
    st.markdown(result_html, unsafe_allow_html=True)
