#----------------------------------------------------------------------------------------------------
try:
    import streamlit as st
    from streamlit import runtime
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    from bs4 import BeautifulSoup
    from typing import Optional, List, Mapping, Any
    import datetime
    import functools
    import hmac
    import logging
    import os.path
    import re
    import requests
    import sys
    import textwrap

    import langchain
    from langchain.chains.question_answering import load_qa_chain
    from langchain.llms.base import LLM
    from langchain.text_splitter import CharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings as SentenceTransformerEmbeddings
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper
    from langchain_community.vectorstores import Qdrant
    from langchain_openai import OpenAI
except:
    print(sys.exc_info())

#-------------------------------------------------------------------
langchain.verbose = False
apikeyfile = '/mnt/sdc1/llm_text_apps/openai_api.txt'
log_filename = 'logs/llm_wrapper.log'
page_name = '03_URL-Loader-LLM'

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
        logging.info("["+page_name+"][check_password]["+get_remote_ip()+"] logged")
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
                "max_tokens": 2048,
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
                "stop": ["Human: ","<|eot_id|>","<|end_of_text|>","Note: "], 
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
@st.cache_data(show_spinner="Fetching data from Wikipedia...")
def fetching_article(wikipediatopic,chunk_size,chunk_overlap):
    wikipage = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    text = wikipage.run(wikipediatopic)

    # Split the text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    chunks = text_splitter.split_text(text)
    #embeddings = SentenceTransformerEmbeddings(model_name='hku-nlp/instructor-large')
    embeddings = SentenceTransformerEmbeddings(model_name="flax-sentence-embeddings/all_datasets_v4_MiniLM-L6")

    # Create in-memory Qdrant instance
    knowledge_base = Qdrant.from_texts(
        chunks,
        embeddings,
        location=":memory:",
        collection_name="doc_chunks",
    )
    return knowledge_base

#-------------------------------------------------------------------
@timeit
@st.cache_data(show_spinner="Fetching data from URL...")
def fetching_url(userinputquery,chunk_size,chunk_overlap):

    page = requests.get(userinputquery)
    soup = BeautifulSoup(page.text, 'html.parser')
    text = soup.get_text()    

    # Split the text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    chunks = text_splitter.split_text(text)
    #embeddings = SentenceTransformerEmbeddings(model_name='hku-nlp/instructor-large')
    embeddings = SentenceTransformerEmbeddings(model_name="flax-sentence-embeddings/all_datasets_v4_MiniLM-L6")

    # Create in-memory Qdrant instance
    knowledge_base = Qdrant.from_texts(
        chunks,
        embeddings,
        location=":memory:",
        collection_name="doc_chunks",
    )
    return knowledge_base

#-------------------------------------------------------------------
@timeit
def prompting_llm(user_question,_knowledge_base,_chain,k_value,llm_used):
    try:
        with st.spinner(text="Prompting LLM..."):
            re_pattern = '\[(.*)\]'
            brackets = re.compile(re_pattern)
            
            try:
                prompt_brackets = brackets.search(user_question).group(0).replace('[','').replace(']','')
            except:
                prompt_brackets = None
                
            if prompt_brackets is None:
                prompt_brackets = user_question
            else:
                user_question = user_question.replace('[','').replace(']','')
                logging.info("["+page_name+"][Prompt]["+get_remote_ip()+"]["+llm_used+"]: Searching only '"+prompt_brackets+"'")

            doc_to_prompt = _knowledge_base.similarity_search(prompt_brackets, k=k_value)
            docs_stats = _knowledge_base.similarity_search_with_score(prompt_brackets, k=k_value)
            
            logging.info("["+page_name+"][Prompt]["+get_remote_ip()+"]["+llm_used+"]: "+user_question)
            for x in range(len(docs_stats)):
                try:
                    content, score = docs_stats[x]
                    logging.info("["+page_name+"][Chunk]["+get_remote_ip()+"]["+str(x)+"]["+str(score)+"]: "+content.page_content.replace("\n","\\n"))
                except:
                    pass
            # Calculating prompt (takes time and can optionally be removed)
            prompt_len = _chain.prompt_length(docs=doc_to_prompt, question=user_question)
            st.write(f"Prompt len: {prompt_len}")
            # if prompt_len > llm.n_ctx:
            #     st.write(
            #         "Prompt length is more than n_ctx. This will likely fail. Increase model's context, reduce chunk's \
            #             sizes or question length, or retrieve less number of docs."
            #     )
            # Grab and print response
            response = _chain.invoke({"input_documents": doc_to_prompt, "question": user_question},return_only_outputs=True).get("output_text")
            logging.info("["+page_name+"][Response]["+get_remote_ip()+"]["+llm_used+"]: "+response.replace("\n","\\n").strip())
            return response
    except:
        logging.warning("["+page_name+"][prompting_llm]["+get_remote_ip()+"]LLM could not be contacted")
        st.error("LLM could not be contacted")
        return "No response from LLM"
    
#-------------------------------------------------------------------
@timeit
def chunk_search(user_question,_knowledge_base,k_value):
    with st.spinner(text="Prompting LLM..."):
        re_pattern = '\[(.*)\]'
        brackets = re.compile(re_pattern)
        
        try:
            prompt_brackets = brackets.search(user_question).group(0).replace('[','').replace(']','')
        except:
            prompt_brackets = None
            
        if prompt_brackets is None:
            prompt_brackets = user_question

        docs_stats = _knowledge_base.similarity_search_with_score(user_question, k=k_value)
        result = '  \n '+datetime.datetime.now().astimezone().isoformat()
        result = result + "  \nPrompt: "+user_question+ "  \n"
        for x in range(len(docs_stats)):
            try:
                result = result + '  \n'+str(x)+' -------------------'
                content, score = docs_stats[x]
                result = result + "  \nContent: "+content.page_content
                result = result + "  \n  \nScore: "+str(score)+"  \n"
            except:
                pass    
        return result

#-------------------------------------------------------------------
def main():

    llm_local = webuiLLM()
    OPENAI_API_KEY = get_file_contents(apikeyfile)
    llm_openai = OpenAI(openai_api_key=OPENAI_API_KEY,model='gpt-3.5-turbo-instruct', max_tokens=1024)

    # Load question answering chain
    chain_local = load_qa_chain(llm_local, chain_type="stuff")
    chain_openai = load_qa_chain(llm_openai, chain_type="stuff")
    chain = chain_local
    llm_used = "local"

    if "Helpful Answer:" in chain.llm_chain.prompt.template:
        chain.llm_chain.prompt.template = (
            f"### Human:{chain.llm_chain.prompt.template}".replace(
                "Helpful Answer:", "\n### Assistant:"
            )
        )
            
#-------------------------------------------------------------------
    # URL page setup
    st.set_page_config(page_title="Ask Wikipedia or URL", layout="wide")
    st.header("Ask Wikipedia or URL 📚")
    userinputquery = st.text_input('Add the desired Wikipedia topic here, or a URL')

    with st.expander("Advanced options"):
        k_value = st.slider('Top K search | default = 6', 2, 30, 6)
        chunk_size = st.slider('Chunk size | default = 1000 [Rebuilds the Vector store]', 500, 1500, 1000, step = 20)
        chunk_overlap = st.slider('Chunk overlap | default = 20 [Rebuilds the Vector store]', 0, 400, 200, step = 20)
        chunk_display = st.checkbox("Display chunk results")
        if get_file_contents(apikeyfile) != 'no_key':
            llm_selection = st.checkbox("Use OpenAI API instead of local LLM - [Faster, but it costs me a little money]")
            if llm_selection:
                chain = chain_openai
                llm_used = "openai"
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
        
    if userinputquery:
        if userinputquery.startswith("http"):
            knowledge_base = fetching_url(userinputquery,chunk_size,chunk_overlap)
        else:
            knowledge_base = fetching_article(userinputquery,chunk_size,chunk_overlap)
       
        user_question = st.text_input("Ask a question about the loaded content. You can use [] to narrow the dataset search.")
        
        promptoption = st.selectbox(
                        '...or select a prompt templates',
                        ("🇺🇸 Summarize the page", "🇧🇷 Faça um resumo da pagina em português"),index=None,
                        placeholder="Select a prompt template...")
        
        if promptoption:
            user_question = promptoption
            
        if user_question:
            response = prompting_llm("This is a page content, based on this text " + user_question.strip(),knowledge_base,chain,k_value,llm_used).replace("\n","  \n")
            st.write("Prompt: _"+user_question.strip()+"_")
            st.write(response)
            if chunk_display:
                chunk_display_result = chunk_search(user_question.strip(),knowledge_base,k_value)
                st.divider()
                with st.expander("Chunk results"):
                    chunk_display_result = '  \n'.join(l for line in chunk_display_result.splitlines() for l in textwrap.wrap(line, width=120))
                    st.code(chunk_display_result)
#-------------------------------------------------------------------

if __name__ == "__main__":
    main() 
