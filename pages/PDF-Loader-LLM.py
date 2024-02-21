import streamlit as st
import requests
import langchain
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings as SentenceTransformerEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms.base import LLM
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
@st.cache_data(hash_funcs={StringIO: StringIO.getvalue},show_spinner="Fetching data from PDF files...")
def fetching_pdf(pdf):
    text = ''
    for f in pdf:
        pdf_reader = PdfReader(f)
        # Iterate through each page in the PDF document to extract the text and add to plain-text string
        for page in pdf_reader.pages:
            text += page.extract_text()

    # Split the text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
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
@st.cache_data(hash_funcs={StringIO: StringIO.getvalue},show_spinner="Prompting LLM...")
def prompting_llm(user_question,_knowledge_base,_chain):
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
    response = _chain({"input_documents": docs, "question": user_question},return_only_outputs=True).get("output_text")
    return response
#-------------------------------------------------------------------
def main():
    # Callback just to stream output to stdout, can be removed
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

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
    # PDF page setup
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
    pdf = st.file_uploader("Upload PDF file(s)", type=["pdf"], accept_multiple_files=True)
    
    if pdf:
        knowledge_base = fetching_pdf(pdf)
        user_question = st.text_input("Ask a question about your PDF:")

        if user_question:
            response = prompting_llm(user_question,knowledge_base,chain)
            st.write(response)
#-------------------------------------------------------------------

if __name__ == "__main__":
    main() 
