from transformers import T5ForConditionalGeneration, T5Tokenizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Load the FLAN-T5 model and tokenizer
    model_name = "google/flan-t5-large"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    logger.info("Model and tokenizer loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model or tokenizer: {e}")
    raise

def generate_question(context):
    """
    Generate an interview question based on the provided context.
    """
    if not isinstance(context, str) or not context.strip():
        logger.error("Invalid context provided.")
        return "Context must be a non-empty string."
    
    try:
        input_text = f"Generate an interview question for a candidate with the following context: {context}"
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50, num_return_sequences=1, temperature=0.7)
        question = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return question
    except Exception as e:
        logger.error(f"Error generating question: {e}")
        return "Failed to generate a question. Please try again."

def evaluate_response(question, response):
    """
    Evaluate the user's response to the interview question.
    """
    if not question or not response:
        logger.error("Invalid question or response provided.")
        return "Both question and response must be non-empty strings."
    
    try:
        input_text = f"Evaluate the following response to the question '{question}': {response}"
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=100, num_return_sequences=1, temperature=0.7)
        evaluation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return evaluation
    except Exception as e:
        logger.error(f"Error evaluating response: {e}")
        return "Failed to evaluate the response. Please try again."
