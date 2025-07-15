import streamlit as st
import requests
import ast
from src.logger import setup_logger

logger = setup_logger(__name__)

API_ASK_URL = "http://localhost:8000/ask"
API_INGEST_URL = "http://localhost:8000/ingest"

st.title("RAG Ingestion & Question Answering")

if "ingested" not in st.session_state:
    st.session_state.ingested = False
if "ingest_status" not in st.session_state:
    st.session_state.ingest_status = ""
if "last_question" not in st.session_state:
    st.session_state.last_question = ""
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

st.header("Step 1: Ingest a Text File")
uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
metadata_str = st.text_area(
    "Optional: Add metadata as a Python dictionary (e.g. {'author': 'John', 'topic': 'AI'})",
    value="{}"
)

if st.button("Ingest"):
    logger.info("Ingest button clicked")
    if uploaded_file is not None:
        try:
            logger.debug("Uploaded file: %s", uploaded_file.name)
            metadata = ast.literal_eval(metadata_str)
            if not isinstance(metadata, dict):
                st.session_state.ingest_status = "Metadata must be a dictionary."
            else:
                files = {"file": (uploaded_file.name, uploaded_file, "text/plain")}
                data = {"metadata": str(metadata)}
                response = requests.post(API_INGEST_URL, files=files, data=data)
                if response.status_code == 200 and response.json().get("status") == "success":
                    st.session_state.ingested = True
                    st.session_state.ingest_status = "Ingestion successful!"
                else:
                    st.session_state.ingest_status = f"Ingestion failed: {response.text}"
        except Exception as e:
            logger.error("Metadata error: %s", e)
            st.session_state.ingest_status = f"Metadata error: {e}"
    else:
        st.session_state.ingest_status = "Please upload a .txt file."

if st.session_state.ingest_status:
    st.info(st.session_state.ingest_status)

# --- Step 2: Ask a Question (always available, but warns if nothing ingested) ---
st.header("Step 2: Ask a Question")
question = st.text_input("Enter your question:", value=st.session_state.last_question)
filter_str = st.text_area(
    "Optional: Add filters as a Python dictionary (e.g. {'author': 'John', 'topic': 'AI'})",
    value="{}"
)
if st.button("Ask"):
    logger.info("Ask button clicked")
    if question.strip():
        logger.debug("Question entered: %s", question)
        st.session_state.last_question = question
        try:
            filters = ast.literal_eval(filter_str)
            if not isinstance(filters, dict):
                st.session_state.last_answer = {"text": "Error: Filters must be a dictionary.", "citations": []}
            else:
                payload = {"query": question, "filters": filters}
                print("Payload for ask request:", payload)
                response = requests.post(API_ASK_URL, json=payload)
                if response.status_code == 200:
                    st.session_state.last_answer = response.json()
                else:
                    st.session_state.last_answer = {"text": "Error: Could not get answer.", "citations": []}
        except Exception as e:
            logger.error("Filter error: %s", e)
            st.session_state.last_answer = {"text": f"Filter error: {e}", "citations": []}

if st.session_state.last_answer:
    st.subheader("Answer")
    st.write(st.session_state.last_answer["text"])
    st.subheader("Citations")
    for citation in st.session_state.last_answer.get("citations", []):
        st.write(citation)

if not st.session_state.ingested:
    st.warning("No data has been ingested yet. Please ingest a text file before asking questions for meaningful results.")
