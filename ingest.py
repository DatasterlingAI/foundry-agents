"""Load the Meridian policy documents into Azure AI Search.

Steps: read Markdown files -> split into sections -> create embeddings
-> (re)create the search index -> upload the chunks.
Runs in GitHub Actions with keyless sign-in (no API keys anywhere).
"""
import os
import pathlib
import re

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField, SearchField, SearchFieldDataType,
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile,
    AzureOpenAIVectorizer, AzureOpenAIVectorizerParameters,
    SemanticSearch, SemanticConfiguration, SemanticPrioritizedFields, SemanticField,
)
from openai import AzureOpenAI

SEARCH_ENDPOINT = os.environ["SEARCH_ENDPOINT"]
AOAI_ENDPOINT = os.environ["AOAI_ENDPOINT"]
EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]
INDEX_NAME = os.environ.get("INDEX_NAME", "meridian-docs")
DOCS_DIR = pathlib.Path(os.environ.get("DOCS_DIR", "data/docs"))
REPO_URL = os.environ.get("REPO_URL", "https://github.com/DatasterlingAI/foundry-agents")
MAX_CHARS = 2000  # keep each chunk small enough to be precise

credential = DefaultAzureCredential()


def split_document(path):
    """Split a Markdown file at '## ' headings so each chunk is one topic."""
    text = path.read_text(encoding="utf-8")
    title_match = re.search(r"^# (.+)$", text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else path.stem
    url = f"{REPO_URL}/blob/main/{DOCS_DIR.as_posix()}/{path.name}"

    parts = re.split(r"^## ", text, flags=re.MULTILINE)
    header, sections = parts[0], parts[1:]
    chunks = []
    for i, section in enumerate(sections or [header]):
        heading = section.partition("\n")[0] if sections else title
        # Repeat the document title and ID in every chunk so each one stands alone
        content = f"{header.strip()}\n\n## {section.strip()}" if sections else header
        for j in range(0, len(content), MAX_CHARS):
            chunks.append({
                "id": f"{path.stem}-{i}-{j // MAX_CHARS}",
                "title": title,
                "section": heading.strip(),
                "content": content[j:j + MAX_CHARS],
                "url": url,
            })
    return chunks


def build_index():
    """Index with a built-in vectorizer: the search service turns each question
    into an embedding itself, using its own managed identity."""
    return SearchIndex(
        name=INDEX_NAME,
        fields=[
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="title", type=SearchFieldDataType.String),
            SearchableField(name="section", type=SearchFieldDataType.String),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SimpleField(name="url", type=SearchFieldDataType.String),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                hidden=True,
                vector_search_dimensions=1536,
                vector_search_profile_name="vector-profile",
            ),
        ],
        vector_search=VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
            vectorizers=[AzureOpenAIVectorizer(
                vectorizer_name="aoai-vectorizer",
                parameters=AzureOpenAIVectorizerParameters(
                    resource_url=AOAI_ENDPOINT,
                    deployment_name=EMBEDDING_MODEL,
                    model_name=EMBEDDING_MODEL,
                ),
            )],
            profiles=[VectorSearchProfile(
                name="vector-profile",
                algorithm_configuration_name="hnsw",
                vectorizer_name="aoai-vectorizer",
            )],
        ),
        semantic_search=SemanticSearch(
            default_configuration_name="default",
            configurations=[SemanticConfiguration(
                name="default",
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="title"),
                    content_fields=[SemanticField(field_name="content")],
                    keywords_fields=[SemanticField(field_name="section")],
                ),
            )],
        ),
    )


def main():
    chunks = [c for p in sorted(DOCS_DIR.glob("*.md")) for c in split_document(p)]
    print(f"Split {len(list(DOCS_DIR.glob('*.md')))} documents into {len(chunks)} chunks")

    aoai = AzureOpenAI(
        azure_endpoint=AOAI_ENDPOINT,
        azure_ad_token_provider=get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"),
        api_version="2024-10-21",
    )
    embeddings = aoai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[f"{c['title']} - {c['section']}\n{c['content']}" for c in chunks],
    )
    for chunk, item in zip(chunks, embeddings.data):
        chunk["embedding"] = item.embedding
    print(f"Created {len(chunks)} embeddings")

    index_client = SearchIndexClient(SEARCH_ENDPOINT, credential)
    try:
        index_client.delete_index(INDEX_NAME)  # start clean so removed docs disappear
    except ResourceNotFoundError:
        pass
    index_client.create_index(build_index())
    print(f"Created index '{INDEX_NAME}'")

    results = SearchClient(SEARCH_ENDPOINT, INDEX_NAME, credential).upload_documents(chunks)
    failed = [r.key for r in results if not r.succeeded]
    if failed:
        raise SystemExit(f"Upload failed for: {failed}")
    print(f"Uploaded {len(chunks)} chunks")


if __name__ == "__main__":
    main()
