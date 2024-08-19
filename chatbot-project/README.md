# "Six the Musical" Chatbot

“Welcome to the Haus, to the Haus of Holbein, ja!”

A RAG-based chatbot meant to help fans of "Six the musical" learn about the characters, the creators of the show, and most importantly, the history (or herstory, if you will) of the characters depicted in the show.

### Getting Started
1. Install [Miniconda3](http://conda.pydata.org/miniconda.html) if you have not done so. After installing, be sure you have closed and then re-opened the terminal window so the changes can take effect.
2. Create a new env: `conda env create -f environment.yml`
3. Activate the env: `conda activate chatbotenv`
4. Enter project directory: `cd chatbot-project`

### Managing Dependencies
1. Activate the env: `conda activate chatbotenv`
2. run `pip install <package>`
3. run `conda env export > environment.yml`

### Build
1. Upload context documents (follow instructions under the "JSON File Format" section).
2. Run the RAG pipeline: `python main.py`

### Using the Chatbot
1. run `fastapi dev app.py`
2. go to http://127.0.0.1:8000/docs

### Testing
1. Enter testing directory: `cd tests`
2. run `pytest`
3. Exit testing directory: `cd ..`

### JSON File Format
When uploading context files, compile them into a single directory and set the "context_directory" in config.yaml to the path of this directory.

All context files must be JSON files structured as a list of dictionaries. Each dictionary in the list must contain the following keys:

title: A string representing the title of the document.
content: A string containing the main text content of the document.

#### Example
Here is an example of the expected JSON structure:
```json
[
    {
        "title": "Document 1",
        "content": "This is the content of the first document."
    },
    {
        "title": "Document 2",
        "content": "This is the content of the second document."
    }
]
```

Make sure your JSON file adheres to this format to ensure proper document processing by the model.
