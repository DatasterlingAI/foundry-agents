"""Ask the agent questions and print answers with their sources.

Uses the QUESTION environment variable if set, otherwise runs the
sample questions in sample_questions.txt.
"""
import os
import pathlib

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

AGENT_NAME = os.environ.get("AGENT_NAME", "meridian-knowledge-copilot")
here = pathlib.Path(__file__).parent

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
    print("=" * 80)
    print(f"Q: {q}\n")
    print(f"A: {response.output_text}\n")

    sources = set()
    for item in response.output:
        for part in getattr(item, "content", None) or []:
            for note in getattr(part, "annotations", None) or []:
                if getattr(note, "type", "") == "url_citation":
                    sources.add(f"{note.title} - {note.url}")
    for s in sorted(sources):
        print(f"   Source: {s}")
print("=" * 80)
