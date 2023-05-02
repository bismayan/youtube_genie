import os

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import urllib.request
import json
import urllib

def set_openai_key(ret_value: bool = False):
    """Reads the Open AI API Key and sets the env variable OPENAI_API_KEY with the value.
    Also optionally return the value if the ret_value arg is true. 

    Args:
        ret_value (bool, optional): Whether to return the key. Defaults to False.

    Returns:
        str: Value of API Key if ret_value is True else None
    """
    if os.environ.get("OPENAI_API_KEY") is None:
        try:
            from openai_key import api_key
            os.environ["OPENAI_API_KEY"] = api_key
        except:
            raise KeyError("OPENAI_API_KEY is not defined")
    if ret_value:
        return api_key
    else:
        return

def get_llm(model_name="text-davinci-003", temp=0.7):
    llm= OpenAI(model_name=model_name, temperature=temp)
    return llm



#change to yours VideoID or change url inparams
def get_video_title(url):
    params = {"format": "json", "url": url}
    url = "https://www.youtube.com/oembed"
    query_string = urllib.parse.urlencode(params)
    url = url + "?" + query_string

    with urllib.request.urlopen(url) as response:
        response_text = response.read()
        data = json.loads(response_text.decode())
    return(data['title'])



def create_db_and_title_from_url(url, chunk_size, chunk_overlap):
    loader = YoutubeLoader.from_youtube_url(url)
    transcript= loader.load()
    title = get_video_title(url)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                    chunk_overlap=chunk_overlap)
    docs= text_splitter.split_documents(transcript)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs,embeddings)
    return db, title



def get_response_to_question(question, db,title, k=4,model_name="text-davinci-003", temp=0.7):
    chunks = db.similarity_search(question,k=k)
    combined_chunks = " ".join(i.page_content for i in chunks)
    chunks_string="\n".join(f"{i+1}.{j.page_content}" for i,j in enumerate(chunks))
    llm = get_llm(model_name=model_name, temp=temp)
    prompt = PromptTemplate(input_variables=["question", "docs", "title"],
                            template="""
                            You are a helpful assistant that that can answer questions about youtube videos 
                            based on the video's transcript.
        
                            Answer the following question: {question}
                            For the Youtube Video with the title: {title}
                            By searching the following video transcript: {docs}
                            
                            Only use the factual information from the transcript to answer the question.
                            
                            If you feel like you don't have enough information to answer the question, say "I don't know".
                            
                            Your answers should be verbose and detailed.
                            """)
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=question, docs=combined_chunks, title=title)
    return response, chunks_string

def generate_response(question, db, title, temp, k=4):
    response, docs = get_response_to_question(question=question, db=db,title=title, k=k, temp=temp)
    resp = f"""
    For the Video Title:
    {title}

    Question Asked
    
    {question}
    
    Model returned the response: 
    
    {response}

    Based on the relevant information bits given below
    
    {docs}
    
    """
    return resp

def main():
    saved_url="placeholder"
    st.set_page_config(
        page_title="🧞🧞 Bismayan's Youtube Genie",
        page_icon="🧞🧞",
        layout="wide",
        initial_sidebar_state="expanded")
    col1, col2 = st.columns([1,3])
    with col1:
        st.image("./logo.png", width=200)
    with col2:
        st.markdown("<h1 style='text-align: left; color: Black;'> Welcome to Bismayan's Youtube Genie </h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: left; color: Black;'>Your AI agent for answering questions about Youtube Videos </h3>", unsafe_allow_html=True)
        
    with st.sidebar:
        st.title("Advanced Config")
        temp= st.slider("Creativity", min_value=0.1, max_value=0.95, step=0.05, value=0.7)
        k= st.slider("Number of Chunks", min_value=1, max_value=4, step=1, value=4)
        chunk_size= st.slider("Size of Chunks", min_value=200, max_value=1500, step=100, value=1000)  
        chunk_overlap= st.slider("Overlap between Chunks", min_value=0, max_value=500, step=50, value=100)    
    
    st.empty()
    col, _ = st.columns([2,3])
    with col:
        url = st.text_input(
            "Enter a Youtube Video URL", value="")
    if url:
        
            if url!=saved_url and len(url)>0:
                db,title = create_db_and_title_from_url(url, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                st.text(f"Loaded Information about the video titled: {title}")
                saved_url = url    
                col, _ = st.columns([2,3])
                with col:
                    question_input = st.text_input("What would you like to know about the Video?")
                    question_button = st.button("Submit Question")
                if question_button:
                    with st.spinner(text="Doing Genie things..."):
                        response, docs = get_response_to_question(question=question_input, db=db,title=title, k=k, temp=temp)
                        print(docs)
                        st.success(response)
                    with st.expander("Relevant Bits of Transcript"):
                        st.text(docs)
                

if __name__ == "__main__":
    st.cache_data.clear()
    set_openai_key()
    main()
