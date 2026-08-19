import json
import os
from uuid import uuid4

SESSION_DIR = "sessions"

os.makedirs(SESSION_DIR, exist_ok=True)


def create_session():

    sid = str(uuid4())

    with open(f"{SESSION_DIR}/{sid}.json", "w") as f:
        json.dump([], f)

    return sid


def save_chat(session_id, messages):

    with open(f"{SESSION_DIR}/{session_id}.json", "w") as f:
        json.dump(messages, f, indent=2)


def load_chat(session_id):

    path = f"{SESSION_DIR}/{session_id}.json"

    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)

    return []
