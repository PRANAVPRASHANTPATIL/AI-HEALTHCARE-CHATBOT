import streamlit as st
from transformers import pipeline
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK data if not already present
nltk.download('punkt')
nltk.download('stopwords')

# Load a healthcare-focused AI model (optional: replace with 'microsoft/BioGPT')
chatbot = pipeline("text-generation", model="distilgpt2")

# Function to process user input and remove stopwords
def process_input(user_input):
    words = word_tokenize(user_input.lower())  # Convert input to lowercase
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(filtered_words)

# Healthcare-specific chatbot logic
def healthcare_chatbot(user_input):
    user_input = process_input(user_input)  # Clean input

    # Rule-based responses
    if "symptom" in user_input:
        return "It seems like you're experiencing symptoms. Please consult a doctor for accurate advice."
    elif "appointment" in user_input:
        return "Would you like me to schedule an appointment with a doctor?"
    elif "medication" in user_input:
        return "It's important to take your prescribed medications regularly. If you have concerns, consult your doctor."
    elif "emergency" in user_input or "urgent" in user_input:
        return "If this is a medical emergency, please call emergency services immediately!"
    elif "diet" in user_input or "nutrition" in user_input:
        return "Maintaining a balanced diet is crucial for good health. Would you like nutrition advice?"
    elif "mental health" in user_input:
        return "Mental health is important! If you're feeling stressed or anxious, talking to a professional might help."
    
    # Generate AI-based response for other queries
    try:
        response = chatbot(user_input, max_length=100, num_return_sequences=1)
        return response[0]['generated_text'].strip()
    except Exception as e:
        return "I'm sorry, I couldn't process that request. Please try again."

# Streamlit web app
def main():
    st.title("AI Healthcare Assistant Chatbot 🤖🏥")

    # Chat history (to maintain context)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for msg in st.session_state.messages:
        st.write(msg)

    # User input box
    user_input = st.text_area("How can I assist you today?", "")

    # Submit button
    if st.button("Submit"):
        if user_input:
            response = healthcare_chatbot(user_input)
            user_message = f"**User:** {user_input}"
            bot_message = f"**Healthcare Assistant:** {response}"

            # Store chat history
            st.session_state.messages.append(user_message)
            st.session_state.messages.append(bot_message)

            # Display response
            st.write(user_message)
            st.write(bot_message)
        else:
            st.write("⚠ Please enter a query.")

if __name__ == "__main__":
    main()
