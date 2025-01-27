from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the FLAN-T5 model and tokenizer
model_name = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def generate_question(context):
    """
    Generate an interview question based on the provided context.
    """
    input_text = f"Generate an interview question for a candidate with the following context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=50, num_return_sequences=1)
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return question

def evaluate_response(question, response):
    """
    Evaluate the user's response to the interview question.
    """
    input_text = f"Evaluate the following response to the question '{question}': {response}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=100, num_return_sequences=1)
    evaluation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return evaluation
