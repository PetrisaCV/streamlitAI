import streamlit as st
import tempfile
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

from transformers import pipeline


# ----------------- CONVERSION TO MARKDOWN -----------------
def convert_to_markdown(file_path: str) -> str:
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        pdf_opts = PdfPipelineOptions(do_ocr=False)
        pdf_opts.accelerator_options = AcceleratorOptions(
            num_threads=4,
            device=AcceleratorDevice.CPU
        )
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pdf_opts,
                    backend=DoclingParseV2DocumentBackend
                )
            }
        )
        doc = converter.convert(file_path).document
        return doc.export_to_markdown(image_mode="placeholder")

    if ext in [".doc", ".docx"]:
        converter = DocumentConverter()
        doc = converter.convert(file_path).document
        return doc.export_to_markdown(image_mode="placeholder")

    if ext == ".txt":
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1", errors="replace")

    raise ValueError(f"Unsupported extension: {ext}")


# ----------------- STREAMLIT APP -----------------
def main():
    st.title("📄 AI-Powered Q&A from Uploaded Documents")

    uploaded_file = st.file_uploader("Upload a document (.pdf, .docx, .txt)", type=["pdf", "doc", "docx", "txt"], key="file_upload")

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        with st.spinner("Converting to Markdown..."):
            markdown_text = convert_to_markdown(tmp_path)
            st.success("✅ Document converted!")

        with st.expander("🔍 View raw Markdown"):
            st.markdown(markdown_text)

        # Chunk the document
        with st.spinner("Splitting into chunks..."):
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = splitter.create_documents([markdown_text])
            st.success(f"✅ Split into {len(chunks)} chunks.")

        # Embed and store chunks
        with st.spinner("Embedding and indexing..."):
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            db = Chroma.from_documents(chunks, embedding=embeddings)
            retriever = db.as_retriever()

        # Load QA pipeline
        with st.spinner("Loading QA model..."):
            hf_pipeline = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", max_new_tokens=500)
            llm = HuggingFacePipeline(pipeline=hf_pipeline)
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
            st.success("✅ Ready to answer questions!")

        # Ask questions
        question = st.text_input("Ask a question about your document:")
        if question:
            with st.spinner("Thinking..."):
                answer = qa_chain.run(question)
            st.markdown(f"**Answer:** {answer}")


if __name__ == "__main__":
    main()
