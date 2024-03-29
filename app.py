import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import io


import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import os
from PIL import Image
import google.generativeai as genai

from dotenv import load_dotenv

load_dotenv() ##load all the environment variables .env
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



def get_gemini_response(input,image,prompt):
    modelI = genai.GenerativeModel('gemini-pro-vision')
    response = modelI.generate_content([input,image[0],prompt])
    return response.text

def input_image_details(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        # Assuming the image will be JPEG or PNG (Modify if needed)
        mime_type = "image/jpeg" 

        image_parts = [
            {
                "mime_type" : mime_type,
                "data" : bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")
    
def compress_image(image_file):  #handle in-memory image data
    image = Image.open(io.BytesIO(image_file.getvalue()))

    quality_start = 99
    quality_step = 5
    target_size = 4 * 1024 * 1024  # 4MB

    while True:
        output_buffer = io.BytesIO()
        image.save(output_buffer, format="JPEG", optimize=True, quality=quality_start)
        filesize = output_buffer.getbuffer().nbytes

        if filesize <= target_size:
            return output_buffer.getvalue()
        else:
            output_buffer.close()
            quality_start -= quality_step

            if quality_start < 0:  
                return None 
            
def image_processing(uploaded_file,user_ques):
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.",use_column_width=True)
    input_prompt = """ 
    You are an expert in understanding images. We will upload a image 
    and you will have to answer any questions based on the uploaded  image
    """
    image_data = input_image_details(uploaded_file)
    response = get_gemini_response(input_prompt,image_data,user_ques)
    return response


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks,embedding = embeddings)
    vector_store.save_local("faiss_index")
    
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer: 
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)
    
    prompt = PromptTemplate(template=prompt_template,input_variables=["context","question"])
    chain = load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain

def pdf_processing(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index",embeddings)
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents":docs, "question":user_question},
        return_only_outputs=True)
    
    return response["output_text"]
    

def main():
    st.title("Multi-Modal Question Answering System 💁")
    content_type = st.radio("Choose input type:", ["Image", "PDF"])
    user_question = st.text_input("Ask your question:",key="input")
    submit = st.button("Process ✅")
    
    if content_type == "Image":
        image_file = st.file_uploader("Upload a single image", type=["jpg", "jpeg", "png"])
        if image_file is not None and user_question and submit:
            with st.spinner("Processing..."):
                try:
                    if image_file.size > (4 * 1024 * 1024):  
                        compressed_image = compress_image(image_file)
                        if compressed_image:
                            response = image_processing(io.BytesIO(compressed_image), user_question) 
                        else:
                            st.error("Unable to compress image below 4MB")
                    else:
                        response = image_processing(image_file, user_question)

                    st.subheader("Response :")
                    st.write(response)

                except Exception as e:  
                    st.error(f"An error occurred: {e}")
        else:
            st.error("Please upload an image")

    
    elif content_type == "PDF":
        pdf_files = st.file_uploader("Upload your PDF files", type="pdf", accept_multiple_files=True)
        
        if pdf_files is not None and user_question and submit:
            with st.spinner("Processing..."):
                try:
                    raw_text = get_pdf_text(pdf_files)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    response = pdf_processing(user_question)
                    st.subheader("Response :")
                    st.write(response)
                except:
                    st.error("Upload a pdf")
        else:
            st.error("Please upload atleast a single pdf")
    else:
            st.error("Please select an input type and upload a file")


if __name__ == "__main__":
    main()
    
