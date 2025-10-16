import torch
import string
from transformers import (BertTokenizer, BertForMaskedLM,
                          XLNetTokenizer, XLNetModel, AutoModelWithLMHead,
                          AutoTokenizer, logging)

logging.set_verbosity_error()

no_words_to_be_predicted = globals()
select_model = globals()
enter_input_text = globals()


def set_model_config(**kwargs):
    for key, value in kwargs.items():
        print("{0} = {1}".format(key, value))

    no_words_to_be_predicted = list(kwargs.values())[0]  # integer values
    select_model = list(kwargs.values())[1]  # possible values = 'bert' or 'gpt' or 'xlnet'
    enter_input_text = list(kwargs.values())[2]  # only string

    return no_words_to_be_predicted, select_model, enter_input_text


def load_model(model_name):
    try:
        if model_name.lower() == "bert":
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()
            return bert_tokenizer, bert_model
        elif model_name.lower() == "gpt":
            gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")
            gpt_model = AutoModelWithLMHead.from_pretrained("gpt2").eval()
            return gpt_tokenizer, gpt_model
        else:
            xlnet_tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
            xlnet_model = AutoModelWithLMHead.from_pretrained("xlnet-base-cased").eval()
            return xlnet_tokenizer, xlnet_model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


def get_all_predictions(input_text, model_name, top_clean):
    tokenizer, model = load_model(model_name)
    generated_text = input_text

    if tokenizer is None or model is None:
        return {'error': 'Model failed to load'}

    if model_name.lower() == "gpt":
        input_ids = tokenizer.encode(generated_text, return_tensors='pt')
        max_length = input_ids.shape[1] + top_clean
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return {'gpt': generated_text}
    elif model_name.lower() == "bert":
        pass
    elif model_name.lower() == "xlnet":
        pass
    else:
        return {'error': 'Unsupported model name'}


def get_prediction_end_of_sentence(input_text, model_name):
    try:
        if model_name.lower() == "bert":
            input_text += ' [mask]'
            print(input_text)
            res = get_all_predictions(input_text, model_name, top_clean=int(no_words_to_be_predicted))
            return res
        elif model_name.lower() == "gpt":
            print('Input: ', input_text)
            res = get_all_predictions(input_text, model_name, top_clean=int(no_words_to_be_predicted))
            return res

    except Exception as error:
        pass


try:
    print("Next Word Prediction with Pytorch using BERT, GPT, and XLNet")
    no_words_to_be_predicted, select_model, enter_input_text = set_model_config(no_words_to_be_predicted=10,
                                                                                select_model="gpt",
                                                                                enter_input_text="why are the neural networks so ")

    res = get_prediction_end_of_sentence(enter_input_text, select_model)
    model_key = select_model.lower()
    print(f"Model: {model_key.upper()}")
    print(f"Generated Text: {res[model_key]}")

except Exception as e:
    print('Some problem occured')
