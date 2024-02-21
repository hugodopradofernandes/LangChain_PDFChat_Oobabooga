# LangChain_Wrapper_LocalLLM

Wrapper to chat with a local llm, sending custom content: Webpages, PDFs, Youtube video transcripts.
_Its a fork from https://github.com/sebaxzero/LangChain_PDFChat_Oobabooga_

Oobabooga install is not covered here!!!

This it just a test using [wafflecomposite/langchain-ask-pdf-local](https://github.com/wafflecomposite/langchain-ask-pdf-local) but with [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) api, all run locally

I added multiple PDF files at once, updated Oobabooga api and raised some limits, also added a Wikipedia query page.

![screenshot](https://github.com/hugodopradofernandes/LangChain_PDFChat_Oobabooga/blob/main/screenshots/Screenshot_20240220_071721.png)
![screenshot](https://github.com/hugodopradofernandes/LangChain_PDFChat_Oobabooga/blob/main/screenshots/Screenshot_20240220_071827.png)
# To install

1. clone repository
2. create a conda environment
3. install the requirements

# To run 

1. start your oobabooga-api # Oobabooga install is not covered here!!!
2. Run start_linux.sh (or run streamlit run app.py)

# Embeddings model

using [flax-sentence-embeddings/all_datasets_v4_MiniLM-L6](https://huggingface.co/flax-sentence-embeddings/all_datasets_v4_MiniLM-L6), will download on first run to the huggingface cache.

# Credits

A fork from https://github.com/sebaxzero/LangChain_PDFChat_Oobabooga

100% not my code, i just copy - paste

main code by [wafflecomposite/langchain-ask-pdf-local](https://github.com/wafflecomposite/langchain-ask-pdf-local)

webui class by [ChobPT/oobaboogas-webui-langchain_agent](https://github.com/ChobPT/oobaboogas-webui-langchain_agent)

conda env files by [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) (i just like using them)
