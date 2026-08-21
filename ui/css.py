import streamlit as st


def load_styles():
    st.markdown(
        """
        <style>

        /* App background */
        .stApp {
            background: #e8f2ef;
        }


        /* Remove left spacing after avatar removal */
        div[data-testid="stChatMessageContent"] {
            margin-left: 0px;
        }


        /* User message - right side */
        .user-msg {
            margin-left: auto;
            margin-right: 0;
            width: fit-content;
            max-width: 70%;
            background-color: #DCF8C6;
            padding: 10px 15px;
            border-radius: 15px;
            text-align: right;
            margin-bottom: 10px;
        }


        /* Assistant message - left side */
        .assistant-msg {
            margin-left: 0;
            margin-right: auto;
            width: fit-content;
            max-width: 70%;
            background-color: #FFFFFF;
            padding: 10px 15px;
            border-radius: 15px;
            text-align: left;
            margin-bottom: 10px;
        }


        </style>
        """,
        unsafe_allow_html=True,
    )