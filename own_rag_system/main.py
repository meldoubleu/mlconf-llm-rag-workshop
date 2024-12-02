from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores import ElasticVectorSearch
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp
from elasticsearch import Elasticsearch
import pdfplumber
import os

# **Schritt 1: PDF-Inhalte extrahieren**
def extract_text_from_pdfs(pdf_folder):
    documents = []
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            with pdfplumber.open(os.path.join(pdf_folder, pdf_file)) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text()
                documents.append({"text": text, "file_name": pdf_file})
    return documents

# **Schritt 2: Indizierung der Dokumente in Elasticsearch**
def index_documents_in_elastic(documents, es_host="http://localhost:9200", index_name="documents"):
    es = Elasticsearch(es_host)
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name)

    for i, doc in enumerate(documents):
        es.index(index=index_name, id=i, body={"content": doc["text"], "file_name": doc["file_name"]})

    print(f"Indiziert {len(documents)} Dokumente in Elasticsearch.")

# **Schritt 3: RAG-Setup mit Llama und Elasticsearch**
def create_rag_qa_pipeline(llm_model_path, es_host="http://localhost:9200", index_name="documents"):
    # Llama LLM
    llama = LlamaCpp(model_path=llm_model_path)
    
    # Elasticsearch als Vector Store
    vector_store = ElasticVectorSearch(
        elasticsearch_url=es_host,
        index_name=index_name,
        embedding=llama
    )
    
    # Retrieval-Augmented QA
    qa = RetrievalQA(llm=llama, retriever=vector_store.as_retriever())
    return qa

# **Schritt 4: Benutzerabfragen**
def chat_with_rag_qa(qa_pipeline):
    print("Frage etwas zu deinen Dokumenten (oder 'exit', um zu beenden):")
    while True:
        query = input("Deine Frage: ")
        if query.lower() == "exit":
            break
        response = qa_pipeline.run(query)
        print("Antwort:", response)

# **Main Script**
if __name__ == "__main__":
    # Pfad zu den PDF-Dokumenten
    pdf_folder = "./PDF_data"

    # Pfad zum Llama-Modell
    llama_model_path = "./models/llama-7b.ggmlv3.q4_0.bin"

    # Elasticsearch-Host und Index-Name
    es_host = "http://localhost:9200"
    index_name = "documents"

    # 1. Dokumente extrahieren und indizieren
    print("Extrahiere und indiziere Dokumente...")
    documents = extract_text_from_pdfs(pdf_folder)
    index_documents_in_elastic(documents, es_host, index_name)

    # 2. RAG-Setup erstellen
    print("Erstelle RAG QA-Pipeline...")
    qa_pipeline = create_rag_qa_pipeline(llama_model_path, es_host, index_name)

    # 3. Chat starten
    print("Starte Chat...")
    chat_with_rag_qa(qa_pipeline)
