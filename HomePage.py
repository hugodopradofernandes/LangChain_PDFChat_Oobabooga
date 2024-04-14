#----------------------------------------------------------------------------------------------------
try:
    import streamlit as st
    from streamlit import runtime
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    from io import StringIO
    from typing import Optional, List, Mapping, Any
    import datetime
    import functools
    import hmac
    import logging
    import os.path
    import requests
    import sys

    import langchain
    from langchain.chains import ConversationChain
    from langchain.chains.conversation.memory import ConversationSummaryMemory
    from langchain.llms.base import LLM
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import OpenAI
    from langchain_openai import OpenAIEmbeddings
except:
    print(sys.exc_info())

#-------------------------------------------------------------------
langchain.verbose = False
apikeyfile = '/mnt/sdc1/llm_text_apps/openai_api.txt'
log_filename = 'logs/llm_wrapper.log'
page_name = 'HomePage'

logging.basicConfig(
filename=log_filename,
format='%(asctime)s %(levelname)-2s %(message)s',
level=logging.INFO,
datefmt='%Y-%m-%d %H:%M:%S')

logging.getLogger('CharacterTextSplitter').disabled = True

#-------------------------------------------------------------------
def get_remote_ip() -> str:
    """Get remote ip."""
    try:
        ctx = get_script_run_ctx()
        if ctx is None:
            return None
        session_info = runtime.get_instance().get_client(ctx.session_id)
        if session_info is None:
            return None
    except Exception as e:
        return "no_IP"
    return session_info.request.remote_ip

#-------------------------------------------------------------------
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        logging.warning("["+page_name+"][check_password]["+get_remote_ip()+"] Password incorrect")
        st.error("😕 Password incorrect")
    return False

if os.path.isfile(".streamlit/secrets.toml"):
    if not check_password():
        st.stop()  # Do not continue if check_password is not True.

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
def timeit(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        start_time = datetime.datetime.now()
        result = func(*args, **kwargs)
        elapsed_time = datetime.datetime.now() - start_time
        logging.info("["+page_name+"][function]["+get_remote_ip()+"][{}] finished in {} ms".format(
            func.__name__, str(elapsed_time)))
        return result
    return new_func

#-------------------------------------------------------------------
def get_file_contents(filename):
    try:
        with open(filename, 'r') as f:
            # It's assumed our file contains a single line,
            # with our API key
            return f.read().strip()
    except FileNotFoundError:
        logging.warning("["+page_name+"][get_file_contents]["+get_remote_ip()+"] OpenAI API key not found - This API won't be available")
        return "no_key"
    
#-------------------------------------------------------------------
@timeit
def prompting_llm(prompt,_chain,llm_used):
    try:
        with st.spinner(text="Prompting LLM..."):
            logging.info("["+page_name+"][Prompt]["+get_remote_ip()+"]["+llm_used+"]: "+prompt)
            response = _chain.invoke(prompt).get("response").replace("\n","  \n")
            logging.info("["+page_name+"][Response]["+get_remote_ip()+"]["+llm_used+"]: "+response.replace("\n","\\n").strip())
            return response
    except:
        logging.warning("["+page_name+"][prompting_llm]["+get_remote_ip()+"]LLM could not be contacted")
        st.error("LLM could not be contacted")
        return "No response from LLM"

#-------------------------------------------------------------------
@timeit
def commands(prompt,last_prompt,last_response,llm_used,chain):
    match prompt.split(" ")[0]:
        case "/continue":
            prompt = "Given this question: " + last_prompt.strip() + ", continue the following text you already started: " + last_response.rsplit("\n\n", 3)[0]
            response = prompting_llm(prompt,chain,llm_used).replace("\n","  \n")
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
            
        case "/recall":
            return "Prompt: _"+last_prompt+"_  \n  \nResponse: "+last_response

        case "/repeat":
            prompt = last_prompt.strip()
            response = prompting_llm(prompt,chain,llm_used).replace("\n","  \n")
            return response
        
        case "/stop":
            headers = {'Accept': 'application/json'}
            r = requests.post('http://127.0.0.1:5000/v1/internal/stop-generation', headers=headers)
            r.raise_for_status()
            if r.status_code == 200:
                return "Ok, generation stopped."
            else:
                return "Stop command failed. Sometimes the LLM API becomes busy while generating text..."
            
        case "/help":
            return "Comand list available: /continue, /history, /list, /load, /model, /recall, /repeat, /stop, /help"

#-------------------------------------------------------------------
def main():

    #Instantiate chat LLM and the search agent
    llm_local = webuiLLM()
    OPENAI_API_KEY = get_file_contents(apikeyfile)
    llm_openai = OpenAI(openai_api_key=OPENAI_API_KEY,model='gpt-3.5-turbo-instruct')

    # Load question answering chain
    chain_local = ConversationChain(llm=llm_local, memory=ConversationSummaryMemory(llm=llm_local,max_token_limit=500), verbose=False)
    chain_openai = ConversationChain(llm=llm_openai, memory=ConversationSummaryMemory(llm=llm_openai,max_token_limit=500), verbose=False)
    chain = chain_local
    llm_used = "local-llm"
    
#-------------------------------------------------------------------
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = []
    #else:
        #chain.memory = st.session_state.history
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
        
#-------------------------------------------------------------------
    # Main page setup
    st.set_page_config(page_title="LLM Wrapper", layout="wide")
    st.header("This is a LLM Wrapper 💬")
    st.info('Select a page on the side menu or use the chat below.', icon="📄")
    if OPENAI_API_KEY != 'no_key':
        with st.expander("Advanced options"):
            llm_selection = st.checkbox("Use OpenAI API instead of local LLM - [Faster, but it costs me a little money]")
            if llm_selection:
                chain = chain_openai
                llm_used = "openai-llm"
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

    st.divider()

#-------------------------------------------------------------------
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
            response = commands(prompt.strip(),last_prompt,last_response,llm_used,chain).replace("\n","  \n")
            # Display assistant response in chat message container
            with st.chat_message("assistant",avatar="🔮"):
                st.markdown(response)
        else:
            response = prompting_llm(prompt.strip(),chain,llm_used)
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Save chat history buffer to the session
    try:
        st.session_state.history = chain.memory
        st.session_state.last_response = response
        if not prompt.startswith("/"):
            st.session_state.last_prompt = prompt
    except:
        pass
    
#-------------------------------------------------------------------

if __name__ == "__main__":
    main() 
