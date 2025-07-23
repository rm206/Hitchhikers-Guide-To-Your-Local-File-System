# Hitchhikers-Guide-To-Your-Local-File-System

A local-first, multimodal semantic search application for your personal files, powered by SentenceTransformers and a local vector database.

## Overview

This project provides a desktop application that allows you to perform semantic searches across files and folders on your local machine. Instead of relying on keyword matching, it understands the *meaning* behind your query to find the most relevant text documents and images.

The entire process, from file indexing to search, runs completely on your machine. Your files and data are never sent to the cloud, ensuring privacy and offline capability.

## Features

-   **Semantic Search**: Finds files based on natural language queries, understanding context and relevance beyond simple keywords.
-   **Multimodal Embeddings**: Uses a single model (`jinaai/jina-clip-v2`) to create a unified embedding space for both text and image content, allowing for cross-modal search.
-   **Local First**: All file processing, embedding generation, and data storage are handled locally using an embedded ChromaDB instance.
-   **GUI Interface**: A simple and clean user interface built with `customtkinter` for selecting directories, entering queries, and viewing results.
-   **Efficient Syncing**: Automatically detects new, modified, and deleted files within a chosen directory to keep the search index up-to-date without re-processing the entire folder on every run.
-   **Direct File Access**: Search results provide direct links to open the file or reveal it in your system's file explorer.

## How It Works

1.  **Select a Folder**: The user selects a root directory to be indexed.
2.  **Indexing**: The application scans the directory for supported files. It compares the filesystem state against its database to identify changes.
3.  **Embedding**:
    -   For text-based documents, content is extracted using `unstructured.io`, chunked, and then converted into vector embeddings.
    -   For images, the file itself is converted directly into a vector embedding.
4.  **Storage**: These embeddings and associated metadata (like file path and modification date) are stored in a local `ChromaDB` database located at `~/lfsss/chroma_db`.
5.  **Searching**:
    -   The user's search query is converted into a vector embedding using the same model.
    -   ChromaDB performs a vector similarity search to find the most relevant file chunks or images.
    -   The application displays the unique file paths corresponding to the top results.

## Supported File Types

The system has been tested with the following file extensions:

-   **Documents**: `.pdf`, `.txt`, `.md`, `.csv`, `.docx`, `.xlsx`
-   **Media**: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.gif`

## Getting Started

These instructions assume you have Python 3.10+ and `uv` installed.

### Installation & Running

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/Hitchhikers-Guide-To-Your-Local-File-System.git](https://github.com/your-username/Hitchhikers-Guide-To-Your-Local-File-System.git)
    cd Hitchhikers-Guide-To-Your-Local-File-System
    ```

2.  **Sync dependencies**: This command creates a virtual environment in a `.venv` directory (if one doesn't already exist) and installs all the required packages from the `requirements.txt` file.
    ```bash
    uv sync
    ```

3.  **Run the application**: This command executes the script within the managed virtual environment without needing to manually activate it first.
    ```bash
    uv run gui.py
    ```
    The initial startup may take a moment as the embedding model is downloaded and loaded into memory.

## Compatibility Note

**This application has only been developed and tested on macOS.** While it uses cross-platform libraries and should theoretically work on Windows and Linux, its behavior on those operating systems has not been verified.

## Technical Stack

-   **Vector Embeddings**: [`jinaai/jina-clip-v2`](https://huggingface.co/jinaai/jina-clip-v2) via [`sentence-transformers`](https://sbert.net/)
-   **Vector Database**: [`chromadb`](https://www.trychroma.com/)
-   **GUI Framework**: [`customtkinter`](https://customtkinter.tomschimansky.com/)
-   **File Parsing**: [`unstructured.io`](https://unstructured.io/)
-   **Dependency Management**: [`uv`](https://docs.astral.sh/uv/)