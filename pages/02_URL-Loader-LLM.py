import streamlit as st
import requests
from bs4 import BeautifulSoup
import langchain
from langchain.llms.base import LLM
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings as SentenceTransformerEmbeddings
from typing import Optional, List, Mapping, Any
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
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
def prompting_llm(user_question,_knowledge_base,_chain,k_value):
    with st.spinner(text="Prompting LLM..."):
        doc_to_prompt = _knowledge_base.similarity_search(user_question, k=k_value)
        docs_stats = _knowledge_base.similarity_search_with_score(user_question, k=k_value)
        print('\n# '+datetime.datetime.now().astimezone().isoformat()+' =====================================================')
        print("Prompt: "+user_question+"\n")
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
        print("-------------------\nResponse: "+response+"\n")
        return response
    
#-------------------------------------------------------------------
def main():

    llm = webuiLLM()

    # Load question answering chain
    chain = load_qa_chain(llm, chain_type="stuff")

    if "Helpful Answer:" in chain.llm_chain.prompt.template:
        chain.llm_chain.prompt.template = (
            f"### Human:{chain.llm_chain.prompt.template}".replace(
                "Helpful Answer:", "\n### Assistant:"
            )
        )
            
#-------------------------------------------------------------------
    # URL page setup
    st.set_page_config(page_title="Ask Wikipedia or URL")
    st.header("Ask Wikipedia or URL 📚")
    userinputquery = st.text_input('Add the desired Wikipedia topic here, or a URL')

    with st.expander("Advanced options"):
        k_value = st.slider('Top K search | default = 6', 2, 10, 6)
        chunk_size = st.slider('Chunk size | default = 1000 [Rebuilds the Vector store]', 500, 1500, 1000, step = 20)
        chunk_overlap = st.slider('Chunk overlap | default = 20 [Rebuilds the Vector store]', 0, 400, 200, step = 20)
        
    if userinputquery:
        if userinputquery.startswith("http"):
            knowledge_base = fetching_url(userinputquery,chunk_size,chunk_overlap)
        else:
            knowledge_base = fetching_article(userinputquery,chunk_size,chunk_overlap)
       
        user_question = st.text_input("Ask a question about the loaded content:")
        
        promptoption = st.selectbox(
                        '...or select a prompt templates',
                        ("🇺🇸 Summarize the page", "🇧🇷 Faça um resumo da pagina em português"),index=None,
                        placeholder="Select a prompt template...")
        
        if promptoption:
            user_question = promptoption
            
        if user_question:
            response = prompting_llm("This is a web page, based on this text " + user_question,knowledge_base,chain,k_value)
            st.write(response)
#-------------------------------------------------------------------

if __name__ == "__main__":
    main() 
