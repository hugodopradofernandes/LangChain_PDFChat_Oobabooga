import streamlit as st
import requests
import langchain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings as SentenceTransformerEmbeddings
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import Formatter
from youtube_transcript_api.formatters import TextFormatter
from io import StringIO
import re

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
def fetching_youtubeid(youtubeid):
    if "youtu" in youtubeid:
        data = re.findall(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", youtubeid)
        youtubeid = data[0]
    return youtubeid

#-------------------------------------------------------------------
@st.cache_data(show_spinner="Fetching data from Youtube...")
def fetching_transcript(youtubeid):
    youtubeid = fetching_youtubeid(youtubeid)

    # retrieve the available transcripts
    #transcript_list = YouTubeTranscriptApi.list_transcripts(youtubeid)

    transcript = YouTubeTranscriptApi.get_transcript(youtubeid, languages=['pt', 'en'])
    
    formatter = TextFormatter()
    text = formatter.format_transcript(transcript)

    # Split the text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
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
def prompting_llm(user_question,_knowledge_base,_chain):
    with st.spinner(text="Prompting LLM..."):
        docs = _knowledge_base.similarity_search(user_question, k=4)
        # Calculating prompt (takes time and can optionally be removed)
        prompt_len = _chain.prompt_length(docs=docs, question=user_question)
        st.write(f"Prompt len: {prompt_len}")
        # if prompt_len > llm.n_ctx:
        #     st.write(
        #         "Prompt length is more than n_ctx. This will likely fail. Increase model's context, reduce chunk's \
        #             sizes or question length, or retrieve less number of docs."
        #     )
        # Grab and print response
        response = _chain.invoke({"input_documents": docs, "question": user_question},return_only_outputs=True).get("output_text")
        return response
#-------------------------------------------------------------------
def parseYoutubeURL(url:str):
   data = re.findall(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
   if data:
       return data[0]
   return ""
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
    # YT page setup
    st.set_page_config(page_title="Ask Youtube Video")
    st.header("Ask Youtube Video 📺")
    youtubeid = st.text_input('Add the desired Youtube video ID or URL here.')
    
    if youtubeid:
        knowledge_base = fetching_transcript(youtubeid)
        user_question = st.text_input("Ask a question about the Youtube video:")
        
        promptoption = st.selectbox(
                        '...or select a prompt templates',
                        ("🇺🇸 Summarize the video", "🇧🇷 Faça um resumo do video em português"),index=None,
                        placeholder="Select a prompt template...")
        
        if promptoption:
            user_question = promptoption
            
        if user_question:
            response = prompting_llm("This is a video transcript, based on this text " + user_question,knowledge_base,chain)
            st.write(response)
#-------------------------------------------------------------------

if __name__ == "__main__":
    main() 
