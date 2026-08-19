import streamlit as st
import requests
from datetime import datetime
import re

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(page_title="Credit Card Chatbot", page_icon="🤖", layout="wide")

BACKEND_URL = "http://localhost:8000/api/v1/query"

# -------------------------
# Sidebar
# -------------------------

# st.sidebar.title("⚙️ Settings")

# backend_url = st.sidebar.text_input("Backend URL", value="http://localhost:8000/chat")

# temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

# top_k = st.sidebar.slider("Retrieved Documents", 1, 10, 5)

st.sidebar.markdown("---")
# st.sidebar.info("""
#     **RAG Chatbot**

#     - Ask questions
#     - View retrieved documents
#     - Powered by your backend
#     """)

# -------------------------
# Session State
# -------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------
# Header
# -------------------------

st.title("🤖 Credit Card RAG Chatbot")
st.caption("Your credit card spend analyzer.")

# -------------------------
# Display Previous Messages
# -------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            sources = message.get("sources", [])

            if sources:
                with st.expander("📚 Sources"):
                    for i, src in enumerate(sources, start=1):
                        st.markdown(f"**{i}. {src.get('title','Document')}**")
                        st.write(src.get("content", ""))
                        st.markdown("---")

# -------------------------
# Chat Input
# -------------------------

prompt = st.chat_input("Ask something...")

if prompt:

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        placeholder = st.empty()

        with st.spinner("Searching knowledge base..."):

            try:
                response = requests.post(
                    BACKEND_URL,
                    json={
                        "question": prompt,
                        # "temperature": temperature,
                        # "top_k": top_k,
                    },
                    timeout=120,
                )

                response.raise_for_status()

                data = response.json()

                answer = data.get("answer", "No answer returned.")

                sources = data.get("sources", [])

                placeholder.markdown(answer)

                if sources:
                    with st.expander("📚 Retrieved Sources", expanded=False):

                        for idx, source in enumerate(sources, start=1):

                            st.markdown(f"### {idx}. {source.get('title','Document')}")

                            if "score" in source:
                                st.caption(f"Similarity: {source['score']:.3f}")

                            st.write(source.get("content", ""))

                            if source.get("metadata"):

                                st.json(source["metadata"])

                            st.markdown("---")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "time": datetime.now().isoformat(),
                    }
                )

            except requests.exceptions.ConnectionError:
                st.error(
                    f"Cannot connect to backend at {BACKEND_URL}. Is the backend running?"
                )

            except requests.exceptions.RequestException as e:
                st.error(f"Backend Error:\n\n{e}")

# ---------------
# stream answer
# ---------------


# def stream_answer(prompt, backend_url):

#     response = requests.post(backend_url, json={"question": prompt}, stream=True)

#     for line in response.iter_lines():

#         if line:
#             yield line.decode("utf-8")


# with st.chat_message("assistant"):

#     response = st.write_stream(stream_answer(prompt, backend_url))


# ---------------
# document upload
# ---------------
uploaded_files = st.sidebar.file_uploader(
    "Upload Documents", type=["pdf"], accept_multiple_files=True
)

if uploaded_files:

    for file in uploaded_files:

        requests.post(
            "http://localhost:8000/upload", files={"file": (file.name, file.getvalue())}
        )

    st.sidebar.success("Documents indexed.")


# ---------------
# Citation Highlighting
# ---------------

# def format_citations(answer):

#     return re.sub(r"\[(\d+)\]", r"<sup style='color:blue'>[\1]</sup>", answer)


# st.markdown(format_citations(answer), unsafe_allow_html=True)


# with st.expander("Sources"):

#     for idx, source in citations.items():
#         st.markdown(f"**[{idx}]** {source}")


# ---------------
# save chat
# ---------------

# save_chat(st.session_state.session_id, st.session_state.messages)

# st.sidebar.write(f"Session: {st.session_state.session_id[:8]}")

if st.sidebar.button("🗑 Clear Chat"):
    st.session_state.messages = []
    st.rerun()
