"""Ask the agent questions and print answers with their sources.

Uses the QUESTION environment variable if set, otherwise runs the
sample questions in sample_questions.txt.
"""
import os
import pathlib
import re

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

AGENT_NAME = os.environ.get("AGENT_NAME", "meridian-knowledge-copilot")
DOCS_DIR = pathlib.Path(os.environ.get("DOCS_DIR", "data/docs"))
REPO_URL = os.environ.get("REPO_URL", "https://github.com/DatasterlingAI/foundry-agents")
here = pathlib.Path(__file__).parent

# Map each document title to its page on GitHub, so sources are clickable links
doc_links = {}
for path in DOCS_DIR.glob("*.md"):
    match = re.search(r"^# (.+)$", path.read_text(encoding="utf-8"), re.MULTILINE)
    if match:
        doc_links[match.group(1).strip()] = f"{REPO_URL}/blob/main/{DOCS_DIR.as_posix()}/{path.name}"

project = AIProjectClient(endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
                          credential=DefaultAzureCredential())
openai = project.get_openai_client()

question = os.environ.get("QUESTION", "").strip()
questions = [question] if question else [
    q.strip() for q in (here / "sample_questions.txt").read_text(encoding="utf-8").splitlines() if q.strip()
]

for q in questions:
    response = openai.responses.create(
        input=q,
        tool_choice="required",  # always search the documents first
        extra_body={"agent_reference": {"name": AGENT_NAME, "type": "agent_reference"}},
    )
    answer = re.sub(r"\s*【[^】]*】", "", response.output_text)  # remove raw citation markers

    sources = set()
    for item in response.output:
        for part in getattr(item, "content", None) or []:
            for note in getattr(part, "annotations", None) or []:
                if getattr(note, "type", "") == "url_citation":
                    sources.add(note.title)

    print("=" * 80)
    print(f"Q: {q}\n")
    print(f"A: {answer}\n")
    for title in sorted(sources):
        print(f"   Source: {title} - {doc_links.get(title, 'link not found')}")
print("=" * 80)
