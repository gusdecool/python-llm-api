from app.config import GEMINI_API_KEY
from pydantic import BaseModel, Field
from typing import Optional
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_litellm import ChatLiteLLM


class ChooserResult(BaseModel):
    action: str = Field(
        ...,
        description="The action to take: 'car_hire_agent', 'weather_agent', 'direct_answer', or 'unsupported'"
    )
    direct_response: Optional[str] = Field(
        None,
        description="The response content if action is 'direct_answer' or 'unsupported'"
    )


def choose_agent(prompt: str) -> ChooserResult:
    if not GEMINI_API_KEY:
        # Fallback if no LLM config: do basic keyword routing
        lower = prompt.lower()
        if "car" in lower or "hire" in lower or "rent" in lower:
            return ChooserResult(action="car_hire_agent")
        elif "weather" in lower or "temp" in lower or "rain" in lower or "forecast" in lower:
            return ChooserResult(action="weather_agent")
        elif "mc2" in lower or "mc^2" in lower or "genius" in lower:
            return ChooserResult(action="direct_answer", direct_response="E=mc² is mass-energy equivalence.")
        else:
            return ChooserResult(action="unsupported", direct_response="I can not do that at the moment")

    llm = ChatLiteLLM(model="gemini/gemini-1.5-flash", api_key=GEMINI_API_KEY, temperature=0)

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
        result = chain.invoke({"prompt": prompt})
        if result.action == "unsupported":
            result.direct_response = "I can not do that at the moment"
        return result
    except Exception:
        return ChooserResult(action="unsupported", direct_response="I can not do that at the moment")
