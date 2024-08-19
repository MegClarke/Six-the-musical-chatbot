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
2. run `pip install *package*`
3. run `conda env export > environment.yml`

### Build
1. Initialize the ChromaDB Vector Store: `python init.py`
2. Run the RAG pipeline: `python main.py`

### Testing
1. Enter testing directory: `cd tests`
2. run `pytest`
3. Exit testing directory: `cd ..`
