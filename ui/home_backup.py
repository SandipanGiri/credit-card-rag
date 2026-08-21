import streamlit as st
import requests
from datetime import datetime
import re
from css import load_styles
import uuid
import base64
from PIL import Image
from io import BytesIO
import base64
import json

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(page_title="Credit Card Chatbot", layout="wide")
load_styles()

BACKEND_URL = "http://localhost:8000/api/v1/query/stream"
# UPLOAD_URL = "http://localhost:8000/api/v1/upload"
UPLOAD_URL = "http://localhost:8000/api/v1/documents"


st.markdown(
    """
    <style>

    /* ---------- Global ---------- */

    .stApp {
        background: #e8f2ef;
    }

    </style>
    """,
    unsafe_allow_html=True,
)
# backgrund color  #f7f8fc
# -------------------------
# Sidebar
# -------------------------


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

if "upload_results" not in st.session_state:
    st.session_state.upload_results = []

if "is_streaming" not in st.session_state:
    st.session_state.is_streaming = False

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "reset_uploader" not in st.session_state:
    st.session_state.reset_uploader = False

# Reset the uploader on the rerun
if st.session_state.reset_uploader:
    st.session_state.reset_uploader = False
    st.rerun()

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
# -------------------------
# Header
# -------------------------

st.title("💳 Credit Card AI Assistant")
st.caption("Your credit card spend analyzer.")

# -------------------------
# Display Previous Messages
# -------------------------

for message in st.session_state.messages:
    # with st.chat_message(message["role"]):
    #     st.markdown(message["content"])

    css_class = "user-msg" if message["role"] == "user" else "assistant-msg"

    st.markdown(
        f"""
            <div class="{css_class}">
                {message["content"]}
            </div>
            """,
        unsafe_allow_html=True,
    )

    if message["role"] == "assistant":

        for img in message.get("images", []):

            try:
                image_bytes = base64.b64decode(img)

                image_bytes = base64.b64decode(img["content"])

                st.image(image, caption=img.get("source_file", "Agent Image"))

            except Exception:
                print("unable to display the images")

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

    st.markdown(
        f"""
        <div class="user-msg">
            {prompt}
        </div>
        """,
        unsafe_allow_html=True,
    )

    placeholder = st.empty()

    full_response = ""

    sources = []

    images = []

    with st.spinner("Thinking..."):
        try:

            with requests.post(
                BACKEND_URL,
                json={
                    "query": prompt,
                    "thread_id": st.session_state.thread_id,
                },
                stream=True,
                timeout=120,
            ) as response:

                response.raise_for_status()

                for chunk in response.iter_lines():

                    if not chunk:
                        continue

                    line = chunk.decode("utf-8")

                    if line.startswith("data:"):

                        payload = json.loads(line.replace("data:", "").strip())

                        # streaming token response
                        if "content" in payload:

                            full_response += payload["content"]

                            placeholder.markdown(
                                f"""
                                <div class="assistant-msg">
                                    {full_response}▌
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        # final response from backend
                        elif payload.get("done"):

                            full_response = payload.get("answer", "")

                            sources = payload.get("sources", [])

                            images = payload.get("images", [])

                            placeholder.markdown(
                                f"""
                                <div class="assistant-msg">
                                    {full_response}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                # Final structured response
                if payload.get("done"):

                    if payload.get("answer"):

                        full_response = payload["answer"]

                    sources = payload.get("sources", [])

                    images = payload.get("images", [])

                    placeholder.markdown(
                        f"""
                        <div class="assistant-msg">
                            {full_response}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # citations + images
                if "sources" in payload:

                    sources = payload["sources"]

                if "images" in payload:

                    images = payload["images"]

                # final answer
                placeholder.markdown(
                    f"""
                    <div class="assistant-msg">
                        {full_response}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # save chat memory

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources,
                    "images": images,
                    "time": datetime.now().isoformat(),
                }
            )

            # display sources

            if sources:

                with st.expander("📚 Retrieved Sources"):

                    for idx, source in enumerate(sources, start=1):

                        st.markdown(f"### {idx}. {source.get('title','Document')}")

                        st.write(source.get("content", ""))

                        st.markdown("---")

            # display images

            for img in images:

                try:

                    if isinstance(img, dict):

                        image_data = img.get("content")

                        caption = img.get("source_file", "Agent Image")

                    else:

                        image_data = img
                        caption = "Agent Image"

                    image_bytes = base64.b64decode(image_data)

                    image = Image.open(BytesIO(image_bytes))

                    st.image(image, caption=caption, use_container_width=True)

                except Exception as e:

                    st.error(f"Unable to display image: {e}")

        except requests.exceptions.ConnectionError:

            st.error("Cannot connect to backend")

        except requests.exceptions.RequestException as e:

            st.error(f"Backend Error: {e}")


# ---------------
# document upload
# ---------------


uploaded_files = st.sidebar.file_uploader(
    "Upload Documents",
    type=["pdf", "docx"],
    accept_multiple_files=True,
    key=f"document_uploader_{st.session_state.uploader_key}",
)


if uploaded_files and st.sidebar.button("Ingest"):
    successful_files = []
    failed_files = []

    with st.sidebar.spinner("Ingesting files..."):
        for file in uploaded_files:
            try:
                files = {
                    "file": (
                        file.name,
                        file.getvalue(),
                        file.type,
                    )
                }

                response = requests.post(
                    UPLOAD_URL,
                    files=files,
                    # timeout=60,
                )

                if response.status_code in [201, 200]:
                    successful_files.append(file.name)
                else:
                    try:
                        print(response.json())
                        error_msg = response.json().get("detail", "File upload failed")
                    except ValueError:
                        error_msg = response.text or "File upload failed"

                    failed_files.append(
                        f"{file.name}: Error {response.status_code} - {error_msg}"
                    )

            except requests.exceptions.ConnectionError:
                failed_files.append(f"{file.name}: Could not connect to the server.")

            except requests.exceptions.Timeout:
                failed_files.append(f"{file.name}: Request timed out.")

            # except Exception as e:
            #     failed_files.append(f"{file.name}: {str(e)}")

    # Display results
    if successful_files:
        st.sidebar.success(f"Successfully uploaded {len(successful_files)} file(s).")

        for filename in successful_files:
            st.sidebar.write(f"✅ {filename}")

    if failed_files:
        st.sidebar.error(f"{len(failed_files)} file(s) failed to upload.")

        for error in failed_files:
            st.sidebar.write(f"❌ {error}")


# ---------------
# Citation Highlighting
# ---------------

# def format_citations(answer):

#     return re.sub(r"\[(\d+)\]", r"<sup style='color:blue'>[\1]</sup>", answer)


# st.markdown(format_citations(answer), unsafe_allow_html=True)


# with st.expander("Sources"):

#     for idx, source in citations.items():
#         st.markdown(f"**[{idx}]** {source}")

if st.sidebar.button("🗑 Clear Chat"):
    st.session_state.messages = []
    st.session_state.uploader_key = 0
    st.rerun()
