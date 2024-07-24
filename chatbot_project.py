import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import warnings
import streamlit as st

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model.eval()

def generate_response(prompt, model, tokenizer, max_length=125, temperature=0.7, top_p=0.9):

    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)
    
    outputs = model.generate(
        inputs, 
        attention_mask=attention_mask,
        max_length=max_length, 
        temperature=temperature, 
        top_p=top_p, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

st.title("GPT-2 Chatbot")

st.write("Chatbot: Hello! How can I help you today?")

if 'conversation' not in st.session_state:
    st.session_state.conversation = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def reset_conversation():
    st.session_state.conversation = ""
    st.session_state.chat_history = []

user_input = st.text_input("You: ", key="input")

if st.button("Send"):
    if user_input.lower() in ["exit", "quit", "bye"]:
        st.write("Chatbot: Goodbye!")
    else:
        try:
            st.session_state.chat_history.append(f"You: {user_input}")
            prompt = "\n".join(st.session_state.chat_history)
            response = generate_response(prompt, model, tokenizer)
            st.session_state.chat_history.append(f"Chatbot: {response}")
            st.session_state.conversation += f"You: {user_input}\nChatbot: {response}\n\n"
            st.rerun() 
        except Exception as e:
            st.write(f"Error: {e}")

st.text_area("Conversation:", st.session_state.conversation, height=1000)

if st.button("Reset Conversation"):
    reset_conversation()
    st.rerun()