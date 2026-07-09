from app.config import GEMINI_API_KEY
from typing import TypedDict, Optional, List, Dict, Any
from datetime import datetime
import os
import httpx
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_litellm import ChatLiteLLM
from langgraph.graph import StateGraph, START, END
from app.config import OPEN_WEATHER_API_KEY
from app.log import get_logger
from app.langfuse import get_langfuse_handler

log = get_logger("weather-agent")

# Define the state shape
class WeatherState(TypedDict):
    prompt: str
    location: Optional[str]
    date: Optional[str]
    missing_fields: List[str]
    next_question: Optional[str]
    weather_data: Optional[Dict[str, Any]]
    final_response: Optional[str]


# Structured output Pydantic schema for parsing parameters
class WeatherExtractedDetails(BaseModel):
    location: Optional[str] = Field(None, description="The city name for weather info, e.g. London, Brisbane, Tokyo")
    date: Optional[str] = Field(None, description="Format: YYYY-MM-DD. Date of the weather forecast.")


# Node 1: Extract Parameters from input prompt
def extract_parameters(state: WeatherState) -> Dict[str, Any]:
    log.info("extracting parameters from input prompt")
    
    # TODO change to dynamic API key chooser when we use dynamic model usage
    if not GEMINI_API_KEY:
        return {}
        
    llm = ChatLiteLLM(model="gemini/gemini-2.5-flash", api_key=GEMINI_API_KEY, temperature=0)
    structured_llm = llm.with_structured_output(WeatherExtractedDetails)
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Extract weather details from the user input. Today's date is {current_date}. If no dates are mentioned, do not guess."),
        ("user", "{prompt}")
    ])
    
    chain = prompt_template | structured_llm
    curr_date = datetime.now().strftime("%Y-%m-%d")
    
    try:
        handler = get_langfuse_handler()
        config = {"callbacks": [handler], "tags": ["weather_agent", "extract_parameters"]} if handler else {}
        extracted = chain.invoke({"prompt": state["prompt"], "current_date": curr_date}, config=config)
    except Exception:
        return {}

    location = extracted.location or state.get("location")
    date = extracted.date or state.get("date")
    
    return {
        "location": location,
        "date": date
    }


# Node 2: Validate extracted parameters
def validate_parameters(state: WeatherState) -> Dict[str, Any]:
    missing = []
    if not state.get("location"):
        missing.append("location")
        
    if missing:
        if not GEMINI_API_KEY:
            question = f"Could you please provide the missing details: {', '.join(missing)}?"
            return {"missing_fields": missing, "next_question": question}

        llm = ChatLiteLLM(model="gemini/gemini-2.5-flash", api_key=GEMINI_API_KEY, temperature=0.2)
        question_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful weather assistant. The user wants to search for weather info, but the location is missing. Ask the user politely to provide the location. Keep it short and friendly."),
        ])
        chain = question_prompt | llm
        try:
            handler = get_langfuse_handler()
            config = {"callbacks": [handler], "tags": ["weather_agent", "validate_parameters"]} if handler else {}
            question = chain.invoke({}, config=config).content
        except Exception:
            question = f"Could you please provide the missing details: {', '.join(missing)}?"
        return {"missing_fields": missing, "next_question": question}
        
    return {"missing_fields": [], "next_question": None}


# Node 3: Search weather (API or Mock)
def search_weather(state: WeatherState) -> Dict[str, Any]:
    location = state["location"]
    
    # Try calling OpenWeatherMap API
    if OPEN_WEATHER_API_KEY:
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={OPEN_WEATHER_API_KEY}&units=metric"
            response = httpx.get(url, timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                return {"weather_data": {
                    "temp": data["main"]["temp"],
                    "feels_like": data["main"]["feels_like"],
                    "condition": data["weather"][0]["description"],
                    "humidity": data["main"]["humidity"],
                    "wind_speed": data["wind"]["speed"],
                    "city": data["name"]
                }}
        except Exception:
            pass
            
    # Fallback to mock weather data TODO remove mock data
    return {"weather_data": {
        "temp": 22.5,
        "feels_like": 21.0,
        "condition": "scattered clouds",
        "humidity": 65,
        "wind_speed": 4.1,
        "city": location
    }}


# Node 4: Synthesize response
def synthesize_response(state: WeatherState) -> Dict[str, Any]:
    weather = state["weather_data"]
    if not GEMINI_API_KEY:
        response = f"The weather in {weather['city']} is currently {weather['condition']} with a temperature of {weather['temp']}°C (feels like {weather['feels_like']}°C), humidity at {weather['humidity']}%, and wind speed at {weather['wind_speed']} m/s."
        return {"final_response": response}

    llm = ChatLiteLLM(model="gemini/gemini-2.5-flash", api_key=GEMINI_API_KEY, temperature=0.2)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly weather assistant. Present the weather data to the user in a clean, helpful markdown summary."),
        ("user", "Weather Data: {weather_data}")
    ])
    chain = prompt_template | llm
    try:
        handler = get_langfuse_handler()
        config = {"callbacks": [handler], "tags": ["weather_agent", "synthesize_response"]} if handler else {}
        response = chain.invoke({"weather_data": str(weather)}, config=config).content
    except Exception:
        response = f"The weather in {weather['city']} is currently {weather['condition']} with a temperature of {weather['temp']}°C (feels like {weather['feels_like']}°C), humidity at {weather['humidity']}%, and wind speed at {weather['wind_speed']} m/s."
    return {"final_response": response}


# Conditional routing logic
def route_after_validation(state: WeatherState):
    if state.get("missing_fields"):
        log.warning("missing info, ask_user")
        return "ask_user"
    return "search"


# Construct the graph
workflow = StateGraph(WeatherState)

workflow.add_node("extract_parameters", extract_parameters)
workflow.add_node("validate_parameters", validate_parameters)
workflow.add_node("search_weather", search_weather)
workflow.add_node("synthesize_response", synthesize_response)

workflow.set_entry_point("extract_parameters")
workflow.add_edge("extract_parameters", "validate_parameters")

workflow.add_conditional_edges(
    "validate_parameters",
    route_after_validation,
    {
        "ask_user": END,
        "search": "search_weather"
    }
)

workflow.add_edge("search_weather", "synthesize_response")
workflow.add_edge("synthesize_response", END)

# Compile graph
weather_agent = workflow.compile()
