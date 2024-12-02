# mlconf-llm-rag-workshop

from PyPDF2 import PdfReader

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

pip install PyPDF2 langchain faiss-cpu transformers pdfplumber
