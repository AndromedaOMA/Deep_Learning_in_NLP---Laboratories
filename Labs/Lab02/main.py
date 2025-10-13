import torch
from transformers import (
    BertTokenizer, BertForMaskedLM,
    AutoTokenizer, AutoModelForCausalLM, logging
)
import string

logging.set_verbosity_error()

# Declare the variables properly
no_words_to_be_predicted = None
select_model = None
enter_input_text = None


def set_model_config(**kwargs):
    for key, value in kwargs.items():
        print(f"{key} = {value}")

    no_words_to_be_predicted = kwargs.get("no_words_to_be_predicted", 5)
    select_model = kwargs.get("select_model", "bert")
    enter_input_text = kwargs.get("enter_input_text", "")

    return no_words_to_be_predicted, select_model, enter_input_text


def load_model(model_name):
    try:
        if model_name.lower() == "bert":
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", force_download=True)
            model = BertForMaskedLM.from_pretrained("bert-base-uncased", force_download=True)
        elif model_name.lower() == "gpt":
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
        else:
            raise ValueError("Unsupported model name (use 'bert' or 'gpt')")
        return tokenizer, model
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to load model: {e}")


def predict_with_bert(tokenizer, model, text, top_k=5):
    if "[MASK]" not in text and "<mask>" not in text:
        text += " [MASK]"
    text = text.replace("<mask>", "[MASK]")

    input_ids = tokenizer.encode(text, return_tensors="pt")
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1]

    with torch.no_grad():
        outputs = model(input_ids)
        predictions = outputs.logits[0, mask_idx, :].squeeze()

    top_k_weights, top_k_indices = torch.topk(predictions, top_k, dim=-1)
    predicted_tokens = [tokenizer.decode(idx).strip() for idx in top_k_indices]
    return predicted_tokens


def predict_with_gpt(tokenizer, model, text, top_k=5):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids)
        next_token_logits = outputs.logits[0, -1, :]

    top_k_weights, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
    predicted_tokens = [tokenizer.decode(idx).strip() for idx in top_k_indices]
    return predicted_tokens


def get_prediction_end_of_sentence(tokenizer, model, input_text, model_name, top_k):
    if model_name.lower() == "bert":
        print(f"Input text: {input_text} <mask>")
        return predict_with_bert(tokenizer, model, input_text, top_k)
    elif model_name.lower() == "gpt":
        print(f"Input text: {input_text}")
        return predict_with_gpt(tokenizer, model, input_text, top_k)
    else:
        raise ValueError("Unsupported model name.")


if __name__ == "__main__":
    print("Next Word Prediction with PyTorch using BERT and GPT")

    no_words_to_be_predicted, select_model, enter_input_text = set_model_config(
        no_words_to_be_predicted=5,
        select_model="bert",
        # select_model="gpt",
        enter_input_text="why are [MASK] people so tired"
    )

    tokenizer, model = load_model(select_model)
    results = get_prediction_end_of_sentence(
        tokenizer, model, enter_input_text, select_model, no_words_to_be_predicted
    )

    print("\nTop predictions:")
    for i, word in enumerate(results, 1):
        print(f"{i}. {word}")
