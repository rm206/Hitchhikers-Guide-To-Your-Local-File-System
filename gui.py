from app import LFSSS
import customtkinter as ctk
import os
import subprocess
import sys
import threading
from tkinter import filedialog

# instantiate the main class once to avoid reloading models on every search.
print("Initializing LFSSS system. This may take a moment.")
lfsss_instance = LFSSS()
print("System ready.")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # window configuration
        self.title("Local File Semantic Search")
        self.geometry("1000x1000")
        ctk.set_appearance_mode("System")

        self.selected_path = ""
        self.last_searched_path = ""

        # UI widget configuration
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)  # The results frame will expand

        # path selection frame
        path_frame = ctk.CTkFrame(self)
        path_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        path_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkButton(
            path_frame, text="Select Folder", width=120, command=self.select_folder
        ).grid(row=0, column=0, padx=10, pady=10)

        self.path_label = ctk.CTkLabel(
            path_frame, text="No folder selected", text_color="gray", anchor="w"
        )
        self.path_label.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        # search query frame
        search_frame = ctk.CTkFrame(self)
        search_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")
        search_frame.grid_columnconfigure(0, weight=1)

        self.query_entry = ctk.CTkEntry(
            search_frame, placeholder_text="Enter your search query..."
        )
        self.query_entry.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.query_entry.bind("<Return>", self.start_search_thread)

        self.n_results_menu = ctk.CTkOptionMenu(
            search_frame, values=[str(i) for i in range(1, 11)]
        )
        self.n_results_menu.set("5")
        self.n_results_menu.grid(row=0, column=1, padx=10, pady=10)

        self.search_button = ctk.CTkButton(
            search_frame, text="Search", command=self.start_search_thread
        )
        self.search_button.grid(row=0, column=2, padx=10, pady=10)

        # results frame
        self.results_frame = ctk.CTkScrollableFrame(self, label_text="Results")
        self.results_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

    def select_folder(self):
        """Opens the OS-native dialog to choose a folder."""
        path = filedialog.askdirectory()
        if path:
            self.selected_path = path
            self.path_label.configure(text=path, text_color="white")

    def start_search_thread(self, event=None):
        """Starts the search in a separate thread to prevent the GUI from freezing."""
        if not self.selected_path:
            self.update_results_display(error="Please select a folder first.")
            return

        query = self.query_entry.get()
        if not query:
            self.update_results_display(error="Please enter a search query.")
            return

        # disable button and show loading state
        self.search_button.configure(state="disabled", text="Searching")
        self.update_results_display(info="Searching, please wait")

        # run the actual search in a thread
        search_thread = threading.Thread(target=self.perform_search, daemon=True)
        search_thread.start()

    def perform_search(self):
        """The core search logic that runs in the background."""
        path = self.selected_path
        query = self.query_entry.get()
        n_results = int(self.n_results_menu.get())

        # if path has changed, prep the DB
        if path != self.last_searched_path:
            print(f"Path changed to '{path}'. Preparing database...")
            lfsss_instance.prep_db_for_search(path)
            self.last_searched_path = path
        else:
            print("Path unchanged. Searching directly.")

        results = lfsss_instance.search(query=query, path=path, n_results=n_results)

        # schedule the UI update to run on the main thread
        self.after(0, self.update_results_display, results)

    def update_results_display(self, results=None, error=None, info=None):
        """Clears old results and displays new ones. Must run on the main thread."""
        # clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        # re-enable search button
        self.search_button.configure(state="normal", text="Search")

        if error:
            ctk.CTkLabel(self.results_frame, text=error, text_color="red").pack(pady=10)
            return

        if info:
            ctk.CTkLabel(self.results_frame, text=info).pack(pady=10)
            return

        if not results or not results.get("metadatas") or not results["metadatas"][0]:
            ctk.CTkLabel(self.results_frame, text="No results found.").pack(pady=10)
            return

        unique_paths = sorted(
            list({m["full_file_path"] for m in results["metadatas"][0]})
        )

        for i, path in enumerate(unique_paths):
            result_item_frame = ctk.CTkFrame(self.results_frame)
            result_item_frame.pack(fill="x", padx=5, pady=5)

            result_item_frame.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(
                result_item_frame, text=path, wraplength=500, justify="left"
            ).grid(row=0, column=0, padx=10, pady=5, sticky="w")

            # use lambda to capture the correct path for each button
            ctk.CTkButton(
                result_item_frame,
                text="Open File",
                width=100,
                command=lambda p=path: self.open_file(p),
            ).grid(row=0, column=1, padx=5, pady=5)

            ctk.CTkButton(
                result_item_frame,
                text="Show in Folder",
                width=120,
                command=lambda p=path: self.show_in_folder(p),
            ).grid(row=0, column=2, padx=5, pady=5)

    def open_file(self, path):
        """Opens the file with the default system application."""
        try:
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":  # macOS
                subprocess.run(["open", path])
            else:  # linux
                subprocess.run(["xdg-open", path])
        except Exception as e:
            print(f"Failed to open file {path}: {e}")

    def show_in_folder(self, path):
        """Reveals the file in the OS file explorer."""
        try:
            if sys.platform == "win32":
                subprocess.run(["explorer", "/select,", os.path.normpath(path)])
            elif sys.platform == "darwin":  # macOS
                subprocess.run(["open", "-R", path])
            else:  # linux
                subprocess.run(["xdg-open", os.path.dirname(path)])
        except Exception as e:
            print(f"Failed to show file {path} in folder: {e}")


if __name__ == "__main__":
    app = App()
    app.mainloop()
