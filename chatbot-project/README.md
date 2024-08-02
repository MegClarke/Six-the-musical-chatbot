# "Six the Musical" Chatbot

“Welcome to the Haus, to the Haus of Holbein, ja!”

A RAG-based chatbot meant to help fans of "Six the musical" learn about the characters, the creators of the show, and most importantly, the history (or herstory, if you will) of the characters depicted in the show. 

### Getting Started
1. Install [Miniconda3](http://conda.pydata.org/miniconda.html) if you have not done so. After installing, be sure you have closed and then re-opened the terminal window so the changes can take effect.
2. Create a new env: `conda env create`
3. Activate the env: `source activate projectvenv`
4. Run `pre-commit install`
5. Please note: If you run this repository with GLAIR-AI-COMMONS repository the precommit will FAIL because pre-commit will make `.git` as base directory path. Please make sure you use this base code in you stand alone repository with all this `base_python_code` in main directory as `.git` file.

### Managing Dependencies
1. Activate the env: `source activate projectvenv`
2. Update `environment.yml`
3. Update the env: `conda env update`

or

1. Activate the env: `source activate projectvenv`
2. run `pip install *package*`
3. run `conda env export > environment.yml`

### Build


#### Prerequisite Files
1. if any
