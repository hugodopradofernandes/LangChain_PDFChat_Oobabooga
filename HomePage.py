import streamlit as st
import requests
import langchain
from langchain.llms.base import LLM
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
from typing import Optional, List, Mapping, Any
from io import StringIO
import datetime
import functools

#-------------------------------------------------------------------
class webuiLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = requests.post(
            "http://127.0.0.1:5000/v1/completions",
            json={
                "prompt": prompt,
                "max_tokens": 512,
                "do_sample": "false",
                "temperature": 0.7,
                "top_p": 0.1,
                "typical_p": 1,
                "repetition_penalty": 1.18,
                "top_k": 40,
                "min_length": 0,
                "no_repeat_ngram_size": 0,
                "num_beams": 1,
                "penalty_alpha": 0,
                "seed": -1,
                "add_bos_token": "true",
                "ban_eos_token": "false",
                "skip_special_tokens": "false",
                "stop": ["Human: "],
            }
        )

        response.raise_for_status()

        return response.json()["choices"][0]["text"].strip().replace("```", " ")
     
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {

        }

#-------------------------------------------------------------------
langchain.verbose = False
#-------------------------------------------------------------------
def timeit(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        start_time = datetime.datetime.now()
        result = func(*args, **kwargs)
        elapsed_time = datetime.datetime.now() - start_time
        print('function [{}] finished in {} ms'.format(
            func.__name__, str(elapsed_time)))
        return result
    return new_func

#-------------------------------------------------------------------
# Main page setup
st.set_page_config(page_title="LLM Wrapper")
st.header("This is a LLM Wrapper 💬")
st.write("Select a page on the side menu or use the chat below")
with st.sidebar.success("Choose a page above"):
    st.sidebar.markdown(
    f"""
    <style>
    [data-testid='stSidebarNav'] > ul {{
        min-height: 40vh;
    }} 
    </style>
    """,
    unsafe_allow_html=True,)

#-------------------------------------------------------------------
#Instantiate chat LLM and the search agent
llm = webuiLLM()
chain = ConversationChain(llm=llm, memory=ConversationSummaryMemory(llm=llm,max_token_limit=500), verbose=False)

#-------------------------------------------------------------------
@timeit
def prompting_llm(prompt,_chain):
    with st.spinner(text="Prompting LLM..."):
        print('\n# '+datetime.datetime.now().astimezone().isoformat()+' =====================================================')
        print("Prompt: "+prompt+"\n")
        response = _chain.invoke(prompt).get("response")
        print("-------------------\nResponse: "+response+"\n")
        return response

#-------------------------------------------------------------------
@timeit
def commands(prompt,last_prompt,last_response):
    match prompt.split(" ")[0]:
        case "/continue":
            prompt = "Given this question: " + last_prompt + ", continue the following text you already started: " + last_response.rsplit("\n\n", 3)[0]
            response = prompting_llm(prompt,chain)
            return response
        
        case "/history":
            try:
                history = chain.memory.load_memory_variables({"history"}).get("history")
                if history == "":
                    return "No history to display"
                else:
                    return "Current History Summary:  \n" + history
            except:
                return "The history was cleared"

        case "/list":
            headers = {'Accept': 'application/json'}
            r = requests.get('http://127.0.0.1:5000/v1/internal/model/list', headers=headers)
            r.raise_for_status()
            return "Model list:  \n" + """{}""".format("  \n".join(r.json()["model_names"][0:]))
            
        case "/model":
            headers = {'Accept': 'application/json'}
            r = requests.get('http://127.0.0.1:5000/v1/internal/model/info', headers=headers)
            r.raise_for_status()
            return "Loaded model:  \n" + r.json()["model_name"]
                   
        case s if s.startswith('/load'):
            model = prompt.split(" ")[1]
            headers = {'Accept': 'application/json'}
            if model in requests.get('http://127.0.0.1:5000/v1/internal/model/list', headers=headers).json()["model_names"][0:]:
                r = requests.post('http://127.0.0.1:5000/v1/internal/model/load', headers=headers, json={"model_name": model})
                r.raise_for_status()
                if r.status_code == 200:
                    return "Ok, model changed."
                else:
                    return "Load command failed."
            else:
                return "Model not in the list. Check the list with the /list command."
            
        case "/repeat":
            return last_response
        
        case "/stop":
            headers = {'Accept': 'application/json'}
            r = requests.post('http://127.0.0.1:5000/v1/internal/stop-generation', headers=headers)
            r.raise_for_status()
            if r.status_code == 200:
                return "Ok, generation stopped."
            else:
                return "Stop command failed. Sometimes the LLM API becomes busy while generating text..."
            
        case "/help":
            return "Comand list available: /continue, /history, /list, /model, /repeat, /stop, /help"

#-------------------------------------------------------------------
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
else:
    chain.memory = st.session_state.history
if "last_response" not in st.session_state:
    st.session_state.last_response = ""
    last_response = ""
else:
    last_response = st.session_state.last_response
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = ""
    last_prompt = ""
else:
    last_prompt = st.session_state.last_prompt
    
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    if prompt.startswith("/"):
        response = commands(prompt,last_prompt,last_response)
        # Display assistant response in chat message container
        with st.chat_message("assistant",avatar="🔮"):
            st.markdown(response)
    else:
        response = prompting_llm(prompt,chain)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
       
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Save chat history buffer to the session
try:
    st.session_state.history = chain.memory
    st.session_state.last_prompt = prompt
    st.session_state.last_response = response
except:
    pass
