import os
import streamlit as st
from dotenv import load_dotenv

from transcript_loader import fetch_transcript
from rag_pipeline import RAGPipeline

load_dotenv(override=True)

st.set_page_config(page_title="YouTube Video Assistant", layout="centered")

st.title("YouTube Video Assistant")
st.write("Extract transcripts from YouTube videos and ask questions using Gemini RAG.")

api_key = os.getenv("GOOGLE_API_KEY", "")

if not api_key:
    st.error("GOOGLE_API_KEY environment variable is not set. Please add it to your .env file.")
    st.stop()

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "stats" not in st.session_state:
    st.session_state.stats = None
if "active_url" not in st.session_state:
    st.session_state.active_url = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

url_input = st.text_input("Enter YouTube Video URL:")

if url_input and url_input != st.session_state.active_url:
    with st.spinner("Processing transcript and building FAISS vector index..."):
        try:
            docs, stats = fetch_transcript(url_input)
            pipeline = RAGPipeline(api_key=api_key)
            pipeline.build_vector_store(docs)

            st.session_state.pipeline = pipeline
            st.session_state.stats = stats
            st.session_state.active_url = url_input
            st.session_state.chat_history = []
            st.success("Video processed successfully!")
        except Exception as err:
            st.error(f"Error: {str(err)}")

if st.session_state.stats:
    stats = st.session_state.stats
    st.divider()
    st.write(f"**Duration:** {stats['duration']} | **Total Words:** {stats['total_words']:,} | **Segments:** {stats['segment_count']}")

    with st.expander("View Transcript Text"):
        st.text_area("Full Transcript", value=stats["full_text"], height=160, disabled=True)

    st.divider()
    st.subheader("Ask Questions")

    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])

    query = st.chat_input("Ask a question about the video transcript...")
    if query:
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching transcript context..."):
                try:
                    result = st.session_state.pipeline.ask(query)
                    st.write(result["answer"])
                    st.session_state.chat_history.append({"question": query, "answer": result["answer"]})
                except Exception as err:
                    st.error(f"Error generating answer: {str(err)}")
