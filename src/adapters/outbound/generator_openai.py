from typing import List
import openai
from ...domain.model import BaseDocument, Answer, Citation
from ...domain.ports import GenerateAnswerPort
from litellm import completion, acompletion
from ...application.prompt import rag_prompt
from src.logger import setup_logger

logger = setup_logger(__name__)

def format_question_and_context(prompt:str, question: str, docs: List[BaseDocument]) -> str:
    context = "\n".join([f"Document {i+1}: {doc.text}" for i, doc in enumerate(docs)])
    return rag_prompt.format(question=question, context=context)

class LitellmGenerator(GenerateAnswerPort):
    def __init__(self, model: str = "openai/gpt-4o-mini") -> None:
        self._model = model
        logger.info("LitellmGenerator initialized with model=%s", model)

    async def generate(self, question: str, docs: List[BaseDocument]) -> Answer:
        logger.info("Generating answer for question: '%s' with %d docs", question, len(docs))
        answer = await acompletion(
            model=self._model,
            messages=[
                {"role": "user", "content": format_question_and_context(rag_prompt, question, docs)}
            ],
            stream=False,
        )
        logger.debug("LLM response: %s", answer.choices[0].message.content)
        return Answer(
            text=answer.choices[0].message.content,
            citations=[]
        )
