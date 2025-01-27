from transformers import pipeline

# Load the DeepSeek model (replace with the actual model name)
deepseek_model = pipeline("text-generation", model="deepseek-model-name")

def generate_question(context):
    """
    Generate an interview question based on the provided context.
    """
    prompt = f"Generate an interview question for a candidate with the following context: {context}"
    question = deepseek_model(prompt, max_length=50, num_return_sequences=1)
    return question[0]['generated_text']

def evaluate_response(question, response):
    """
    Evaluate the user's response to the interview question.
    """
    prompt = f"Evaluate the following response to the question '{question}': {response}"
    evaluation = deepseek_model(prompt, max_length=100, num_return_sequences=1)
    return evaluation[0]['generated_text']
