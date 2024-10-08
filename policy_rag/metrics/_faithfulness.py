from dotenv import load_dotenv
load_dotenv()
import json
from typing import List

from langsmith.schemas import Example, Run
from pydantic import BaseModel, Field

from langchain_core.documents.base import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticToolsParser


class Propositions(BaseModel):
    """Use to record each factual assertion."""

    propositions: List[str] = Field(description="The factual propositions generated by the model")


class FaithfulnessScore(BaseModel):
    """Use to score how faithful the propositions are to the docs."""

    reasoning: str = Field(description="The reasoning for the faithfulness score")
    score: bool


def extract_propositions(text: str) -> List[str]:
    template = """
    Extract all factual statements from the following Text:

    Text:
    {text}
    """
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template(template)

    tools = [Propositions]

    chain = (
        prompt
        | llm.bind_tools(tools)
        | PydanticToolsParser(tools=tools)
    )

    return chain.invoke({'text': text})[0].propositions


def get_faithfulness_score(proposition: str, formatted_docs: str) -> List[str]:
    template = """
    Grade whether the Proposition can be logically concluded
    from the Docs:
    
    Proposition: {proposition}
    
    Docs:
    {formatted_docs}
    """
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template(template)

    tools = [FaithfulnessScore]

    chain = (
        prompt
        | llm.bind_tools(tools)
        | PydanticToolsParser(tools=tools)
    )

    res = chain.invoke({'proposition': proposition, 'formatted_docs': formatted_docs})
    score = res[0].score
    reasoning = res[0].reasoning
    
    return score, reasoning


def faithfulness(run: Run, example: Example) -> dict:
    # Assumes your RAG app includes the prediction in the "output" key in its response
    response: str = run.outputs["answer"].content
    # Assumes your RAG app includes the retrieved docs as a "context" key in the outputs
    # If not, you can fetch from the child_runs of the run object
    retrieved_docs: List[Document] = run.outputs["contexts"]
    formatted_docs = "\n".join([doc.page_content for doc in retrieved_docs])
    
    propositions = extract_propositions(response)
    
    scores, reasoning = [], []
    for proposition in propositions:
        score, reason = get_faithfulness_score(proposition, formatted_docs)
        scores.append(score)
        reasoning.append(reason)

    average_score = sum(scores) / len(scores) if scores else None
    comment = "\n".join(reasoning)
    
    return {"key": "faithfulness", "score": average_score, "comment": comment}