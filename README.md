# Multi-Modal Question Answering System

This Python project leverages Streamlit, the Google Generative AI API, and other powerful libraries to build a comprehensive question-answering system. The system can intelligently process both images and PDF documents to answer user questions.

**Key Features**

* **Image Question Answering:**  Upload an image, and the system will use the Google Generative AI (Gemini) model to provide a detailed answer to your image-related query.
* **PDF Question Answering:**  Upload one or multiple PDF documents. The system will process the text, embed it for semantic understanding, and answer your questions based on the PDF's content.

**Setup**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/project-name
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv env 
   source env/bin/activate  # Linux/macOS
   env\Scripts\activate.bat  # Windows 
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt 
   ```

4. **Obtain Google API Key:**
   - Follow the instructions from Google to obtain your Generative AI API Key.

5. **Set Environment Variable:**
   ```bash
   export GOOGLE_API_KEY=your_api_key 
   ```

**Usage**

1. **Start the Streamlit app:**
   ```bash
   streamlit run app.py 
   ```

2. **Use the Web Interface:**
   * Choose whether you want to process an image or a PDF file.
   * Upload your image or PDF document(s).
   * Enter your question in the text box.
   * Click the "Process" button to obtain your answer.

**Dependencies**

* streamlit
* PyPDF2
* langchain
* google.generativeai
* dotenv
* PIL (Pillow)
* FAISS

**Code Structure**

* **app.py:**  Contains the core Streamlit application logic and structure.
* **image_processing() function:** Handles image-based question answering, including image compression and API calls to the Gemini model.
* **pdf_processing() function:** Handles PDF-based question answering, including text extraction, embedding,  vector store creation, and query processing through a conversational chain.

**License**

This project is licensed under the MIT License - see the LICENSE file for details.


