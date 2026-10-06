import streamlit as st
import requests

# Backend API endpoint
API_URL = "http://127.0.0.1:8000/api/chat"

# Configure the Streamlit page aesthetics
st.set_page_config(
    page_title="TN Agri Policy Chatbot",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# App Header
st.title("🌾 TN Agri Policy Chatbot")
st.markdown("""
Welcome! I am an AI assistant trained on the **Tamil Nadu Agriculture Policy Notes 2024-25**. 
Ask me about schemes, budgets, crops, or policy guidelines.
""")
st.divider()

# Initialize session state to store chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with the agricultural policies today?"}]

# Display past chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Field
if prompt := st.chat_input("E.g., What is the scheme for paddy crops?"):
    
    # 1. Display user's prompt in the UI and save to state
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Call the FastAPI backend
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Searching policy documents..."):
            try:
                # Include the chat history in the payload
                payload = {
                    "question": prompt,
                    "chat_history": st.session_state.messages
                }

                # Send POST request to FastAPI
                response = requests.post(API_URL, json=payload)
                response.raise_for_status() # Raise exception for HTTP errors
                
                # Parse the answer
                data = response.json()
                bot_reply = data.get("answer", "I couldn't process that request.")
                
            except requests.exceptions.ConnectionError:
                bot_reply = "⚠️ Error: Cannot connect to the backend. Is FastAPI running?"
            except Exception as e:
                bot_reply = f"⚠️ An error occurred: {str(e)}"
        
        # Display the response
        message_placeholder.markdown(bot_reply)
        
    # 3. Save assistant's reply to state
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})