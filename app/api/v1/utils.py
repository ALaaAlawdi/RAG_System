import os
import uuid


def save_upload(content: bytes, filename: str, directory: str) -> str:
    stem, ext = os.path.splitext(filename)
    unique_name = f"{stem}_{uuid.uuid4().hex[:8]}{ext}"
    path = os.path.join(directory, unique_name)
    with open(path, "wb") as f:
        f.write(content)
    return path