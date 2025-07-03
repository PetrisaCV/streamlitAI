import streamlit as st
import tempfile
from pathlib import Path
import os
import openai
client = openai.OpenAI(api_key="your key")
# The API key is read from the environment variable "OPENAI_API_KEY".
# Set it in your terminal before running the app:
# export OPENAI_API_KEY=your-key-here
# For example, on macOS/Linux:
#   export OPENAI_API_KEY="your key"

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

# ----------------- DOCUMENT MANAGER -----------------
def document_manager():
    if 'documents' not in st.session_state:
        st.session_state.documents = []

    uploaded_file = st.file_uploader("Uncover the secrets in your documents (.pdf, .docx, .txt)", type=["pdf", "doc", "docx", "txt"])
    if uploaded_file:
        # Save file to a temporary location
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        # Store file info in session_state
        st.session_state.documents.append({
            "name": uploaded_file.name,
            "path": temp_path
        })
        st.success(f"Uploaded: {uploaded_file.name}")

    st.write("## Uploaded Documents")
    docs_to_remove = []
    for idx, doc in enumerate(st.session_state.documents):
        col1, col2 = st.columns([4, 1])
        col1.write(doc["name"])
        if col2.button("Delete", key=f"delete_{doc['name']}_{idx}"):
            docs_to_remove.append(idx)

    # Remove selected docs after iteration
    for idx in sorted(docs_to_remove, reverse=True):
        del st.session_state.documents[idx]
        st.experimental_rerun()

# ----------------- DOCUMENT STATS -----------------

def show_document_stats(uploaded_files, all_texts):
    st.subheader("🎶Document Statistics📊🎶")

    if not uploaded_files or not all_texts:
        st.info("No documents to analyze.")
        return

    total_docs = len(uploaded_files)
    total_words = 0
    file_types = {}

    for i, uploaded_file in enumerate(uploaded_files):
        ext = Path(uploaded_file.name).suffix.lower()
        file_types[ext] = file_types.get(ext, 0) + 1
        content = all_texts[i]
        total_words += len(content.split())

    avg_words = total_words // total_docs if total_docs > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("How many documents?", total_docs)
    col2.metric("How many words?", f"{total_words:,}")
    col3.metric("How many words per doc?", f"{avg_words:,}")

    st.write("**File Types:**")
    for ext, count in file_types.items():
        st.write(f"• {ext}: {count} file(s)")

# ----------------- STREAMLIT APP -----------------
def ask_openai(question, context):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ],
        max_tokens=500,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

# ----------------- CUSTOM CSS -----------------
def add_custom_css():
    # Render music notes at the very top of the page for proper z-index stacking
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Comic+Neue:wght@700&display=swap');

        html, body, [class^="appview-container"], [class^="block-container"], .stApp { 
            font-family: 'Comic Neue', cursive !important;
            background: linear-gradient(135deg, #8e2de2, #4a00e0) !important;
            background-color: #8e2de2 !important; /* fallback solid purple */
            color: white !important;
            overflow-x: hidden;
        }

        .st-emotion-cache-1v0mbdj, .st-emotion-cache-18ni7ap {
            background: transparent !important;
        }

        .main-header {
            font-size: 3rem;
            color: #ffe600;
            text-align: center;
            margin-bottom: 2rem;
            padding: 1rem;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        }

        .stButton > button {
            background-color: #ffe600 !important;
            color: #4a00e0 !important;
            border: none !important;
            border-radius: 10px !important;
            font-size: 1.1rem !important;
            font-weight: bold !important;
            padding: 0.75rem 1.5rem !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
            transition: 0.3s ease !important;
        }

        .stButton > button:hover {
            background-color: #fffa65 !important;
            transform: scale(1.05) !important;
            box-shadow: 0 6px 12px rgba(0,0,0,0.3) !important;
        }

        .metric-card {
            background: rgba(255, 255, 255, 0.1);
            padding: 1rem;
            border-radius: 15px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
            text-align: center;
            color: white;
        }

        .music-note {
            position: fixed;
            top: 0;
            font-size: 2rem;
            animation: floatUp 8s infinite linear;
            opacity: 0.8;
            z-index: 99999;
            color: #fff700;
            pointer-events: none;
        }

        @keyframes floatUp {
            0% { transform: translateY(100vh) rotate(0deg); opacity: 0; }
            50% { opacity: 1; }
            100% { transform: translateY(-10vh) rotate(360deg); opacity: 0; }
        }

        .note1 { left: 10%; animation-delay: 0s; }
        .note2 { left: 30%; animation-delay: 2s; }
        .note3 { left: 50%; animation-delay: 4s; }
        .note4 { left: 70%; animation-delay: 1s; }
        .note5 { left: 90%; animation-delay: 3s; }
        </style>
        <div class="music-note note1">🎵</div>
        <div class="music-note note2">🎶</div>
        <div class="music-note note3">🎼</div>
        <div class="music-note note4">🎵</div>
        <div class="music-note note5">🎶</div>
        """, unsafe_allow_html=True)

# ----------------- ENHANCED QUESTION INTERFACE -----------------
def enhanced_question_interface():
    """Professional question asking interface"""
    
    st.subheader("💬 Ask anything about soundtracks! 🎶")
    
    with st.expander("💡 Don't know where to start?"):
        st.markdown("""
        - What are the main topics covered in these documents?  
        - Summarize the key points from [document name]  
        - What does the document say about [specific topic]?  
        - Compare information between documents  
        - Find specific data or statistics  
        """)
    
    question = st.text_input(
        "Ask away:",
        placeholder="e.g., Who is Hans Zimmer?"
    )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        search_button = st.button("🔍 Search my Documents", type="primary")
    with col2:
        clear_button = st.button("🗑️ Clear Search History")

    return question, search_button, clear_button

# Loading animations
def show_loading_animation(text="Thinking and humming Hakuna Matata..."):
    """Show professional loading animation"""
    with st.spinner(text):
        import time
        time.sleep(0.5)  # Brief pause for better UX

# Example usage:
# show_loading_animation("Loading your data...")

# ----------------- MAIN APP FUNCTION -----------------

def main():
    add_custom_css()


    st.title("📄 Petrisa's Soundtrack search engine 🎶")

    tab1, tab2, tab3 = st.tabs(["📁 Upload & Convert", "❓ Ask Questions", "📊 Doc Stats"])

    if "doc" not in st.session_state:
        st.session_state.doc = []

    all_texts = []
    for doc in st.session_state.doc:
        with open(doc["path"], "r", encoding="utf-8") as f:
            all_texts.append(f.read())

# ----------------- TAB 1: Upload & Convert -----------------

    with tab1:
        uploaded_files = st.file_uploader(
        "Upload documents (.pdf, .docx, .txt)",
        type=["pdf", "doc", "docx", "txt"],
        accept_multiple_files=True,
        key="file_upload_tab1"
    )

        if uploaded_files:
            all_texts = []

            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                with st.spinner(f"Converting {uploaded_file.name} to Markdown..."):
                    try:
                        md_text = convert_to_markdown(tmp_path)
                        all_texts.append(md_text)
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")

            markdown_text = "\n\n".join(all_texts)
            st.success(f"✅ {len(uploaded_files)} documents converted.")

            # Track results for better UX
            converted_docs = []
            errors = []

            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    word_count = len(all_texts[i].split())
                    converted_docs.append({
                        "filename": uploaded_file.name,
                        "word_count": word_count
                    })
                except Exception as e:
                    errors.append(f"{uploaded_file.name}: {e}")

            # Show conversion results
            if converted_docs:
                st.write("### Result of markdown conversion:")
                for doc in converted_docs:
                    st.write(f"**{doc['filename']}**: {doc['word_count']} words")
            if errors:
                st.error("Errors occurred during conversion:")
                for err in errors:
                    st.error(err)

            with st.expander("🔍 View raw Markdown"):
                st.markdown(markdown_text)

            with st.spinner("Splitting into chunks..."):
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                chunks = []
                for i, text in enumerate(all_texts):
                    split = splitter.create_documents([text])
                    for chunk in split:
                        chunk.metadata["source"] = uploaded_files[i].name
                    chunks.extend(split)
                st.success(f"✅ Split into {len(chunks)} chunks.")

            with st.spinner("Embedding and indexing..."):
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                db = Chroma.from_documents(chunks, embedding=embeddings)
                retriever = db.as_retriever()

            st.success("✅ Ready to answer all of your questions!")
            st.session_state.retriever = retriever

    def show_search_history():
        if "search_history" in st.session_state and st.session_state.search_history:
            with st.expander("🕑 Search History"):
                for i, entry in enumerate(reversed(st.session_state.search_history)):
                    st.markdown(f"**Q:** {entry['question']}")
                    st.markdown(f"**A:** {entry['answer']}")
                    st.markdown(f"**Sources:** {', '.join(entry['sources'])}")
                    st.markdown("---")
        else:
            st.info("You haven't asked anything yet.")

# ----------------- TAB 2: Ask Questions -----------------

    with tab2:
        if "retriever" not in st.session_state:
            st.warning("Please upload documents first.")
        else:
            retriever = st.session_state.retriever
        question, search_button, clear_button = enhanced_question_interface()
    if search_button and question:
        with st.spinner("Thinking and singing Hakuna Matata..."):
            docs = retriever.get_relevant_documents(question)
            context = "\n\n".join([doc.page_content for doc in docs])
            sources = list({doc.metadata.get("source", "Unknown") for doc in docs})

            answer = ask_openai(question, context)

            if "search_history" not in st.session_state:
                st.session_state.search_history = []

            st.session_state.search_history.append({
                "question": question,
                "answer": answer,
                "sources": sources
            })

            st.markdown(f"**Answer:** {answer}  \n\n**Sources:** {', '.join(sources)}")

    if clear_button:
        st.session_state.search_history = []

    # 🔥 Always show search history
    show_search_history()

    # ----------------- TAB 3: Document Stats -----------------

    with tab3:
        show_document_stats(uploaded_files, all_texts)

        uploaded_files = st.file_uploader(
        "Upload documents (.pdf, .docx, .txt)",
        type=["pdf", "doc", "docx", "txt"],
        accept_multiple_files=True,
        key="file_upload_tab3"
    )

    if uploaded_files:
        all_texts = []

        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            with st.spinner(f"Converting {uploaded_file.name} to Markdown..."):
                try:
                    md_text = convert_to_markdown(tmp_path)
                    all_texts.append(md_text)
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")

        markdown_text = "\n\n".join(all_texts)
        st.success(f"✅ {len(uploaded_files)} documents converted.")

        with st.expander("🔍 View raw Markdown"):
            st.markdown(markdown_text)

        # Chunk the document
        with st.spinner("Splitting into chunks..."):
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = []
            for i, text in enumerate(all_texts):
                split = splitter.create_documents([text])
                # Attach metadata to each chunk
                for chunk in split:
                    chunk.metadata["source"] = uploaded_files[i].name
                chunks.extend(split)
            st.success(f"✅ Split into {len(chunks)} chunks.")

        # Embed and store chunks
        with st.spinner("Embedding and indexing..."):
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # this is a class and the same for each embedding!!
            # do i need to calculate embeddings for my chunks??
            db = Chroma.from_documents(chunks, embedding=embeddings) #poglej dokumentacijo funkcije
            retriever = db.as_retriever()

        # Load QA pipeline with gemma-2b.Q4_K_M.gguf model using llama-cpp-python

        st.success("✅ Ready to answer questions!")

        # Ask questions
        if "search_history" not in st.session_state:
            st.session_state.search_history = []

        def show_search_history():
            if st.session_state.search_history:
                with st.expander("🕑 Search History"):
                    for i, entry in enumerate(reversed(st.session_state.search_history)):
                        st.markdown(f"**Q:** {entry['question']}")
                        st.markdown(f"**A:** {entry['answer']}")
                        st.markdown(f"**Sources:** {', '.join(entry['sources'])}")
                        st.markdown("---")
            else:
                st.info("No search history yet.")

        question = st.text_input("Ask a question about your document:")
        if question:
            with st.spinner("Thinking..."):
                # Retrieve relevant context from the document
                docs = retriever.get_relevant_documents(question)
                context = "\n\n".join([doc.page_content for doc in docs])
                sources = list({doc.metadata.get("source", "Unknown") for doc in docs})

                # Use the OpenAI Chat API to answer the question
                answer = ask_openai(question, context)

                # Save to search history
                st.session_state.search_history.append({
                    "question": question,
                    "answer": answer,
                    "sources": sources
                })

                st.markdown(f"**Answer:** {answer}  \n\n**Sources:** {', '.join(sources)}")

        #show_search_history()
        #show_document_stats(uploaded_files, all_texts)


if __name__ == "__main__":
    main()
