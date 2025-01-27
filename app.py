import streamlit as st
from deepseek_model import generate_question, evaluate_response

# Streamlit App Title
st.title("AI Interviewer 🤖")

# Input for user context
context = st.text_input("Enter your context (e.g., job role, skills):")

if context:
    # Generate an interview question
    question = generate_question(context)
    st.write(f"**Interview Question:** {question}")

    # Input for user's response
    response = st.text_input("Your Answer:")

    if response:
        # Evaluate the response
        evaluation = evaluate_response(question, response)
        st.write(f"**Evaluation:** {evaluation}")
