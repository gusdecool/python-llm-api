from app.config import GEMINI_API_KEY
from pydantic import BaseModel, Field
from typing import Optional
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_litellm import ChatLiteLLM
from app.log import get_logger
from app.langfuse import get_langfuse_handler
from app.util import get_session_id


class ChooserResult(BaseModel):
    action: str = Field(
        ...,
        description="The action to take: 'car_hire_agent', 'weather_agent', 'direct_answer', or 'unsupported'"
    )
    direct_response: Optional[str] = Field(
        None,
        description="The response content if action is 'direct_answer' or 'unsupported'"
    )


log = get_logger('agent-chooser')


def choose_agent(prompt: str) -> ChooserResult:
    log.info('chosing agent')
    llm = ChatLiteLLM(model="gemini/gemini-2.5-flash", api_key=GEMINI_API_KEY, temperature=0)

    structured_llm = llm.with_structured_output(ChooserResult)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an routing agent. Analyze the user prompt and decide the action:\n"
            "- If the user asks for car hire, car rental, or vehicle booking, set action to 'car_hire_agent'.\n"
            "- If the user asks about the weather, temperature, rain, or weather forecast, set action to 'weather_agent'.\n"
            "- If the user asks a general knowledge question that you can answer directly (e.g. 'what is E=mc2'), set action to 'direct_answer' and write the answer in direct_response.\n"
            "- If the user asks for anything else that requires a tool we do not have (e.g. booking a flight, ordering food, writing code), set action to 'unsupported' and set direct_response to exactly 'I can not do that at the moment'."
        )),
        ("user", "{prompt}")
    ])

    chain = prompt_template | structured_llm
    try:
        handler = get_langfuse_handler()
        config = {
            "callbacks": [handler],
            "metadata": {
                "langfuse_session_id": get_session_id(),
                "langfuse_tags": ["agent_chooser", "choose_agent"]
            }
        } if handler else {}
        result = chain.invoke({"prompt": prompt}, config=config)
        if result.action == "unsupported":
            log.info("Iam dumb, I dont know")
            result.direct_response = "I can not do that at the moment"
        return result
    except Exception:
        return ChooserResult(action="unsupported", direct_response="I can not do that at the moment")
