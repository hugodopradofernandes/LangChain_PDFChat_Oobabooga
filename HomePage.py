import streamlit as st
import requests
import langchain
from langchain.llms.base import LLM
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from typing import Optional, List, Mapping, Any
from io import StringIO

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
                "max_tokens": 1024,
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

# Callback just to stream output to stdout, can be removed
callbacks = [StreamingStdOutCallbackHandler()]

#-------------------------------------------------------------------
#Instantiate chat LLM and the search agent
llm = webuiLLM()
chain = ConversationChain(llm=llm, memory=ConversationSummaryMemory(llm=llm,max_token_limit=500), verbose=False)

#-------------------------------------------------------------------
@st.cache_data(show_spinner="Prompting LLM...")
def prompting_llm(prompt,_chain):
    response = _chain.invoke(prompt).get("response")
    return response

#-------------------------------------------------------------------
def commands(prompt):
    match prompt.split(" ")[0]:
        case "/history":
            return "Current History Summary:  \n" + chain.memory.load_memory_variables({"history"}).get("history")
        case "/model":
            headers = {'Accept': 'application/json'}
            r = requests.get('http://127.0.0.1:5000/v1/internal/model/info', headers=headers)
            r.raise_for_status()
            return "Loaded model:  \n" + r.json()["model_name"]
        case "/list":
            headers = {'Accept': 'application/json'}
            r = requests.get('http://127.0.0.1:5000/v1/internal/model/list', headers=headers)
            r.raise_for_status()
            return "Model list:  \n" + """{}""".format("  \n".join(r.json()["model_names"][0:]))
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
        case "/stop":
            headers = {'Accept': 'application/json'}
            r = requests.post('http://127.0.0.1:5000/v1/internal/stop-generation', headers=headers)
            r.raise_for_status()
            if r.status_code == 200:
                return "Ok, generation stopped."
            else:
                return "Stop command failed. Sometimes the LLM API becomes busy while generating text..."
            
        case "/help":
            return "Comand list available: /history, /list, /stop, /help"

#-------------------------------------------------------------------
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "buffer" not in st.session_state:
    st.session_state.buffer = []
else:
    chain.memory = st.session_state.buffer

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
        response = commands(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant",avatar="🔮"):
            st.markdown(response)
    else:
        response = prompting_llm(prompt,chain)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
#         
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Save chat history buffer to the session
st.session_state.buffer = chain.memory
