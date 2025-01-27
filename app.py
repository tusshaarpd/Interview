import streamlit as st
from deepseek_model import generate_question, evaluate_response

# Streamlit App Title
st.title("AI Interviewer 🤖")
st.write("Generate interview questions and evaluate your responses using AI!")

# Input for user context
st.subheader("Provide Context")
context = st.text_input("Enter your context (e.g., job role, skills):", placeholder="e.g., Data Scientist, Python, Machine Learning")

if context:
    # Generate an interview question
    with st.spinner("Generating question..."):
        question = generate_question(context)

    if question.startswith("Failed"):
        st.error("Unable to generate a question. Please try again.")
    else:
        st.success("Question generated successfully!")
        st.write(f"**Interview Question:** {question}")

        # Input for user's response
        st.subheader("Your Response")
        response = st.text_input("Enter your answer to the question:", placeholder="Type your answer here.")

        if response:
            # Evaluate the response
            with st.spinner("Evaluating response..."):
                evaluation = evaluate_response(question, response)

            if evaluation.startswith("Failed"):
                st.error("Unable to evaluate the response. Please try again.")
            else:
                st.success("Response evaluated successfully!")
                st.write(f"**Evaluation:** {evaluation}")

# Optional Reset Button
if st.button("Reset"):
    st.rerun()

