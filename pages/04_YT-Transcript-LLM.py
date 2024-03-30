import streamlit as st
from streamlit import runtime
from streamlit.runtime.scriptrunner import get_script_run_ctx

from io import StringIO
from typing import Optional, List, Mapping, Any
import datetime
import functools
import re
import requests
import textwrap

import langchain
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms.base import LLM
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings as SentenceTransformerEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import Formatter
from youtube_transcript_api.formatters import TextFormatter

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
apikeyfile = '/mnt/sdc1/llm_text_apps/openai_api.txt'
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
@timeit
def get_file_contents(filename):
    try:
        with open(filename, 'r') as f:
            # It's assumed our file contains a single line,
            # with our API key
            return f.read().strip()
    except FileNotFoundError:
        print("OpenAI API key not found - This API won't be available")
        return "no_key"
    
#-------------------------------------------------------------------
@timeit
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
        return None
    return session_info.request.remote_ip

#-------------------------------------------------------------------
@timeit
def fetching_youtubeid(youtubeid):
    if "youtu" in youtubeid:
        data = re.findall(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", youtubeid)
        youtubeid = data[0]
    return youtubeid

#-------------------------------------------------------------------
@timeit
@st.cache_data(show_spinner="Fetching data from Youtube...")
def fetching_transcript(youtubeid,chunk_size,chunk_overlap):
    youtubeid = fetching_youtubeid(youtubeid)

    # retrieve the available transcripts
    #transcript_list = YouTubeTranscriptApi.list_transcripts(youtubeid)

    transcript = YouTubeTranscriptApi.get_transcript(youtubeid, languages=['pt', 'en'])
    
    formatter = TextFormatter()
    text = formatter.format_transcript(transcript)

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
    with st.spinner(text="Prompting LLM..."):
        doc_to_prompt = _knowledge_base.similarity_search(user_question, k=k_value)
        docs_stats = _knowledge_base.similarity_search_with_score(user_question, k=k_value)
        print('\n# '+datetime.datetime.now().astimezone().isoformat()+' from ['+get_remote_ip()+'] =====================================================')
        print("Prompt ["+llm_used+"]: "+user_question+"\n")
        for x in range(len(docs_stats)):
            try:
                print('# '+str(x)+' -------------------')
                content, score = docs_stats[x]
                print("Content: "+content.page_content)
                print("\nScore: "+str(score)+"\n")
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
        print("-------------------\nResponse ["+llm_used+"]:\n"+response+"\n")
        return response

#-------------------------------------------------------------------
@timeit
def chunk_search(user_question,_knowledge_base,k_value):
    with st.spinner(text="Prompting LLM..."):
        doc_to_prompt = _knowledge_base.similarity_search(user_question, k=k_value)
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
@timeit
def parseYoutubeURL(url:str):
   data = re.findall(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
   if data:
       return data[0]
   return ""
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
    # YT page setup
    st.set_page_config(page_title="Ask Youtube Video", layout="wide")
    st.header("Ask Youtube Video 📺")
    youtubeid = st.text_input('Add the desired Youtube video ID or URL here.')

    with st.expander("Advanced options"):
        k_value = st.slider('Top K search | default = 6', 2, 10, 6)
        chunk_size = st.slider('Chunk size | default = 1000 [Rebuilds the Vector store]', 500, 1500, 1000, step = 20)
        chunk_overlap = st.slider('Chunk overlap | default = 20 [Rebuilds the Vector store]', 0, 400, 200, step = 20)
        chunk_display = st.checkbox("Display chunk results")
        if get_file_contents(apikeyfile) != 'no_key':
            llm_selection = st.checkbox("Use OpenAI API instead of local LLM - [Faster, but it costs me a little money]")
            if llm_selection:
                chain = chain_openai
                llm_used = "openai"
        
    if youtubeid:
        knowledge_base = fetching_transcript(youtubeid,chunk_size,chunk_overlap)
        user_question = st.text_input("Ask a question about the Youtube video:")
        
        promptoption = st.selectbox(
                        '...or select a prompt templates',
                        ("🇺🇸 Summarize the video", "🇧🇷 Faça um resumo do video em português"),index=None,
                        placeholder="Select a prompt template...")
        
        if promptoption:
            user_question = promptoption
            
        if user_question:
            response = prompting_llm("This is a video transcript, based on this text " + user_question.strip(),knowledge_base,chain,k_value,llm_used).replace("\n","  \n")
            st.write("_"+user_question.strip()+"_")
            st.write(response)
            if chunk_display:
                chunk_display_result = chunk_search(user_question.strip(),knowledge_base,k_value)
                with st.expander("Chunk results"):
                    chunk_display_result = '  \n'.join(l for line in chunk_display_result.splitlines() for l in textwrap.wrap(line, width=120))
                    st.code(chunk_display_result)
#-------------------------------------------------------------------

if __name__ == "__main__":
    main() 
