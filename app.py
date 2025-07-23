import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from datetime import datetime
import hashlib
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from typing import List
from unstructured.partition.auto import partition


class LFSSS:
    def __init__(self):
        self.FIXED_CHROMA_DB_PATH = os.path.expanduser("~/lfsss/chroma_db")
        self.FIXED_CHROMA_COLLECTION_NAME = "all_files"
        self.RETRIEVAL_MODEL_NAME = "jinaai/jina-clip-v2"

        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.retrieval_model = self._setup_embedding_model()
        self.collection = self._setup_chroma()

    def _setup_embedding_model(self) -> SentenceTransformer:
        """
        Set up the SentenceTransformer model for embeddings.

        Returns:
            SentenceTransformer: The SentenceTransformer model for generating embeddings.
        """
        return SentenceTransformer(self.RETRIEVAL_MODEL_NAME, trust_remote_code=True)

    def _setup_chroma(self) -> chromadb.Collection:
        """
        Set up ChromaDB client and collection.

        This collection will be used to store all embedded files and the associated metadata.

        Returns:
            chromadb.Collection: The ChromaDB collection for storing embedded files.
        """

        class CustomEmbeddingFunction(EmbeddingFunction):
            """
            A custom embedding function for ChromaDB that uses a SentenceTransformer model to generate embeddings for documents.
            Will be passed pre-chunked documents.
            """

            def __init__(self, lfsss_instance):
                self.retrieval_model = lfsss_instance.retrieval_model
                self.device = lfsss_instance.device

            def __call__(self, docs: Documents) -> Embeddings:
                # embed the documents
                # will pass pre-chunked documents
                # will pass filename for images since that works directly with the encode function
                return self.retrieval_model.encode(
                    docs,
                    batch_size=16,
                    normalize_embeddings=True,
                    show_progress_bar=True,
                    device=self.device,
                )

        os.makedirs(self.FIXED_CHROMA_DB_PATH, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=self.FIXED_CHROMA_DB_PATH)
        chroma_collection = chroma_client.get_or_create_collection(
            name=self.FIXED_CHROMA_COLLECTION_NAME,
            metadata={
                "description": "Collection for all embedded files",
                "created": str(datetime.now()),
            },
            embedding_function=CustomEmbeddingFunction(self),
        )
        return chroma_collection

    def _get_filesystem_state(self, path: str) -> List[dict]:
        """
        Returns a list of files in the specified directory with their last modified timestamps.

        Args:
            path (str): The directory path to scan for files.
        Returns:
            list: A list of files with their last modified timestamps.
        """
        files_in_path = []

        if not os.path.isdir(path):
            print(f"Error: Provided path '{path}' is not a valid directory.")
            return files_in_path

        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            for file in files:
                if not file.startswith("."):
                    full_path = os.path.join(root, file)
                    try:
                        mod_time = os.path.getmtime(full_path)
                        files_in_path.append(
                            {"full_file_path": full_path, "last_modified": mod_time}
                        )
                    except OSError as e:
                        print(f"Error accessing file {full_path}: {e}")

        return files_in_path

    def _get_db_state(self, path: str) -> List[dict]:
        """
        Get the current state of the database for a specific path.

        Args:
            path (str): The directory path to check in the database.

        Returns:
            list: A list of files with their last modified timestamps from the database.
        """
        all_entries = self.collection.get(include=["metadatas"])
        matching_entries = []
        for i, metadata in enumerate(all_entries["metadatas"]):
            if (
                metadata
                and "full_file_path" in metadata
                and metadata["full_file_path"].startswith(path)
            ):
                entry_id = all_entries["ids"][i]
                matching_entries.append(
                    {
                        "full_file_path": metadata["full_file_path"],
                        "last_modified": metadata["last_modified"],
                        "id": entry_id,
                    }
                )

        return matching_entries

    def _find_deleted_ids(
        self, filesystem_files: List[dict], db_files: List[dict]
    ) -> List[str]:
        """
        Find IDs of files that are in the database but not in the filesystem.
        Args:
            filesystem_files (list): List of files in the filesystem with their last modified timestamps.
            db_files (list): List of files in the database with their last modified timestamps.
        Returns:
            list: A list of IDs of files that are in the database but not in the filesystem
        """
        # if there's nothing in the database, there's nothing to delete
        if not db_files:
            return []

        # Convert the lists of dictionaries into pandas DataFrames
        db_df = pd.DataFrame(db_files)
        fs_df = pd.DataFrame(filesystem_files)

        # if the filesystem is empty, all files in the database are considered stale
        if fs_df.empty:
            return db_df["id"].tolist()

        # create a boolean mask to identify which 'full_file_path' in the database DataFrame
        # is NOT present in the filesystem DataFrame's 'full_file_path' column
        is_stale_mask = ~db_df["full_file_path"].isin(fs_df["full_file_path"])

        # use the mask to select the stale rows from the database DataFrame
        # and retrieve their corresponding 'id's as a list
        deleted_ids = db_df[is_stale_mask]["id"].tolist()

        return deleted_ids

    def _find_stale_embeddings(
        self, filesystem_files: List[dict], db_files: List[dict]
    ) -> List[str]:
        """
        Find IDs of files that are in the database but not in the filesystem.

        Args:
            filesystem_files (list): List of files in the filesystem with their last modified timestamps.
            db_files (list): List of files in the database with their last modified timestamps.

        Returns:
            list: A list of IDs of files that are in the database but have been modified since
            their last embedding.
        """
        if not db_files or not filesystem_files:
            return []

        db_df = pd.DataFrame(db_files)
        fs_df = pd.DataFrame(filesystem_files)

        # merge the two DataFrames on 'full_file_path' to find common files
        # suffixes are added to distinguish between the 'last_modified' columns
        merged_df = pd.merge(db_df, fs_df, on="full_file_path", suffixes=("_db", "_fs"))

        # if the merge is empty, no common files exist to be stale.
        if merged_df.empty:
            return []

        # create a boolean mask to find rows where the database timestamp is
        # older than the filesystem timestamp. these are the stale files.
        stale_mask = merged_df["last_modified_db"] < merged_df["last_modified_fs"]

        # get the unique file paths of all files identified as stale
        stale_file_paths = merged_df[stale_mask]["full_file_path"].unique()

        if len(stale_file_paths) == 0:
            return []

        # from the original database DataFrame, find all entries (and their IDs)
        # that match the stale file paths. this correctly handles cases where
        # one file has multiple database entries.
        ids_to_delete_mask = db_df["full_file_path"].isin(stale_file_paths)
        stale_ids = db_df[ids_to_delete_mask]["id"].tolist()

        return stale_ids

    def _find_files_to_add(
        self, filesystem_files: List[dict], db_files: List[dict]
    ) -> List[dict]:
        """
        Find files that need to be added to the database based on the filesystem state and the database
        state.
        Args:
            filesystem_files (list): List of files in the filesystem with their last modified timestamps.
            db_files (list): List of files in the database with their last modified timestamps.
        Returns:
            list: A list of files that need to be added to the database.
        """
        if not filesystem_files:
            return []

        fs_df = pd.DataFrame(filesystem_files)

        if not db_files:
            return fs_df.to_dict("records")

        db_df = pd.DataFrame(db_files)

        # first find stale files (modified since last embedding)
        merged_df = pd.merge(fs_df, db_df, on="full_file_path", suffixes=("_fs", "_db"))
        stale_mask = merged_df["last_modified_fs"] > merged_df["last_modified_db"]
        stale_files_df = fs_df[
            fs_df["full_file_path"].isin(merged_df[stale_mask]["full_file_path"])
        ]

        # find new files (in filesystem but not in database)
        fs_paths = fs_df["full_file_path"]
        db_paths = set(db_df["full_file_path"])
        is_new_mask = ~fs_paths.isin(db_paths)
        new_files_df = fs_df[is_new_mask]

        files_to_add_df = pd.concat([new_files_df, stale_files_df]).drop_duplicates()
        return files_to_add_df.to_dict("records")

    def _add_files_to_db(self, files_to_add: List[str]) -> None:
        """
        Add files to the database.

        Args:
            files_to_add (list): A list of files to add to the database.

        Returns:
            None
        """
        all_documents = []
        all_ids = []
        all_metadatas = []

        TEXT_CHUNK_MAX_CHARS = 24000
        IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}

        for file_info in files_to_add:
            file_path = file_info["full_file_path"]
            mod_time = file_info["last_modified"]

            directory_path = os.path.dirname(file_path)
            filename = os.path.basename(file_path)

            try:
                file_extension = os.path.splitext(file_path)[1].lower()

                base_metadata = {
                    "full_file_path": file_path,
                    "directory_path": directory_path,
                    "filename": filename,
                    "last_modified": mod_time,
                }

                if file_extension in IMAGE_EXTENSIONS:
                    document_content = file_path
                    doc_id = hashlib.sha256(document_content.encode()).hexdigest()

                    all_documents.append(document_content)
                    all_ids.append(doc_id)
                    all_metadatas.append(base_metadata)
                else:
                    elements = partition(filename=file_path)
                    chunks_for_file = []
                    current_chunk = ""
                    for el in elements:
                        element_text = el.text
                        if (
                            len(current_chunk) + len(element_text)
                            > TEXT_CHUNK_MAX_CHARS
                        ):
                            if current_chunk:
                                chunks_for_file.append(current_chunk)
                            current_chunk = element_text
                        else:
                            current_chunk += f" {element_text}"

                    if current_chunk:
                        chunks_for_file.append(current_chunk)
                    if not chunks_for_file:
                        continue

                    ids_for_file = []
                    metadatas_for_file = []

                    for chunk_text in chunks_for_file:
                        chunk_hash = hashlib.sha256(chunk_text.encode()).hexdigest()
                        ids_for_file.append(f"{file_path}-{chunk_hash}")
                        metadatas_for_file.append(base_metadata)

                    all_documents.extend(chunks_for_file)
                    all_ids.extend(ids_for_file)
                    all_metadatas.extend(metadatas_for_file)

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue

        if all_documents:
            self.collection.add(
                documents=all_documents, ids=all_ids, metadatas=all_metadatas
            )
            print(f"Added {len(all_documents)} documents to the database.")
        else:
            print("No valid documents to add to the database.")

    def prep_db_for_search(self, path: str) -> None:
        """
        Prepare the database for search by syncing the filesystem state with the database state.

        Args:
            path (str): The directory path to sync with the database.
        Returns:
            None
        """
        filesystem_files = self._get_filesystem_state(path)
        db_files = self._get_db_state(path)

        deleted_ids = self._find_deleted_ids(filesystem_files, db_files)
        stale_ids = self._find_stale_embeddings(filesystem_files, db_files)

        to_delete_ids = list(set(deleted_ids + stale_ids))
        print(
            f"Deleting {len(to_delete_ids)} stale or deleted files from the database."
        )

        if to_delete_ids:
            self.collection.delete(ids=to_delete_ids)

        files_to_add = self._find_files_to_add(filesystem_files, db_files)
        self._add_files_to_db(files_to_add)

    def search(self, query: str, path: str, n_results: int) -> dict:
        """
        Search the database for files matching the query in the specified path.
        Args:
            query (str): The search query.
            path (str): The directory path to search in.
            n_results (int): The number of search results to return.
        Returns:
            dict: The search results containing documents and metadata.
        """
        search_path = os.path.abspath(os.path.expanduser(path))

        if not os.path.isdir(search_path):
            print(f"Error: Provided search path '{path}' is not a valid directory.")
            return {}

        # get all directories in the search path to filter by
        paths_to_search = [search_path]
        for root, dirs, _ in os.walk(search_path):
            for d in dirs:
                paths_to_search.append(os.path.join(root, d))

        where_clause = {"directory_path": {"$in": paths_to_search}}

        query_embedding = self.retrieval_model.encode(
            query,
            prompt_name="retrieval.query",
            normalize_embeddings=True,
        )

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause,
            include=["documents", "metadatas"],
        )

        return results
