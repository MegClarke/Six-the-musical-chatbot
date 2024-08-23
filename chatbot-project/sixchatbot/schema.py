"""Input Validation Schema for config."""


from pydantic import BaseModel


class LlmSchema(BaseModel):
    """Schema for defining the configuration of the language model (LLM).

    Attributes:
        model (str): The name or identifier of the LLM model to be used.
        prompt (str): The prompt template or initial text for the model.
        temp (float): The temperature setting for the model, controlling the randomness of the output.
    """

    model: str
    prompt: str
    temp: float


class RerankerSchema(BaseModel):
    """Schema for defining the configuration of the reranker model.

    Attributes:
        model (str): The name or identifier of the reranker model to be used.
    """

    model: str


class ChromaSchema(BaseModel):
    """Schema for defining the configuration of Chroma, typically used for vector storage.

    Attributes:
        persist_directory (str): The directory path where the Chroma data should be persisted.
    """

    persist_directory: str
    embedding_model: str


class TextSplitterSchema(BaseModel):
    """Schema for defining the configuration of the text splitter.

    Attributes:
        chunk_size (int): The size of the text chunks to be generated.
        chunk_overlap (int): The amount of overlap between consecutive text chunks.
    """

    chunk_size: int
    chunk_overlap: int


class Config(BaseModel):
    """Main configuration schema combining various sub-schemas.

    Attributes:
        context_directory (str): The directory path where the context data files are stored.
        llm (LlmSchema): The configuration settings for the LLM.
        search_kwargs (dict[str, int]): A dictionary of search parameters with string keys and integer values.
        chroma (ChromaSchema): The configuration settings for Chroma.
        text_splitter (TextSplitterSchema): The configuration settings for the text splitter.
    """

    context_directory: str
    llm: LlmSchema
    reranker: RerankerSchema
    search_kwargs: dict[str, int]
    chroma: ChromaSchema
    text_splitter: TextSplitterSchema
