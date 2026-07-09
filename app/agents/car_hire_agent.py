from typing import TypedDict, Optional, List, Dict, Any
from datetime import datetime
import os
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_litellm import ChatLiteLLM
from langgraph.graph import StateGraph, START, END
from app.config import GEMINI_API_KEY


# Define the state shape
class CarHireState(TypedDict):
    prompt: str
    location: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    missing_fields: List[str]
    next_question: Optional[str]
    scraped_deals: Optional[List[Dict[str, Any]]]
    final_response: Optional[str]


# Structured output Pydantic schema for parsing parameters
class ExtractedDetails(BaseModel):
    location: Optional[str] = Field(None, description="The city or airport code for pick-up, e.g. Brisbane, Sydney")
    start_date: Optional[str] = Field(None, description="Format: YYYY-MM-DD. Start/pick-up date of the rental.")
    end_date: Optional[str] = Field(None, description="Format: YYYY-MM-DD. End/drop-off date of the rental.")


# Node 1: Extract Parameters from input prompt
def extract_parameters(state: CarHireState) -> Dict[str, Any]:
    if not GEMINI_API_KEY:
        return {}
        
    llm = ChatLiteLLM(model="gemini/gemini-1.5-flash", temperature=0)
    structured_llm = llm.with_structured_output(ExtractedDetails)
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Extract car hire details from the user input. Today's date is {current_date}. If no dates are mentioned, do not guess."),
        ("user", "{prompt}")
    ])
    
    chain = prompt_template | structured_llm
    curr_date = datetime.now().strftime("%Y-%m-%d")
    
    try:
        extracted = chain.invoke({"prompt": state["prompt"], "current_date": curr_date})
    except Exception as e:
        # Fallback if API fails or is unconfigured
        return {}

    # Merge extracted details with existing state details
    location = extracted.location or state.get("location")
    start_date = extracted.start_date or state.get("start_date")
    end_date = extracted.end_date or state.get("end_date")
    
    return {
        "location": location,
        "start_date": start_date,
        "end_date": end_date
    }


# Node 2: Validate extracted parameters
def validate_parameters(state: CarHireState) -> Dict[str, Any]:
    missing = []
    if not state.get("location"):
        missing.append("location")
    if not state.get("start_date"):
        missing.append("start_date")
    if not state.get("end_date"):
        missing.append("end_date")
        
    if missing:
        if GEMINI_API_KEY:
            question = f"Could you please provide the missing details: {', '.join(missing)}?"
            return {"missing_fields": missing, "next_question": question}

        llm = ChatLiteLLM(model="gemini/gemini-1.5-flash", temperature=0.2)
        question_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful car rental assistant. The user wants to search for car hire, but some fields are missing: {missing}. Ask the user politely to provide the missing information. Do not ask for fields that are not in the list. Keep it short and friendly."),
        ])
        chain = question_prompt | llm
        try:
            question = chain.invoke({"missing": ", ".join(missing)}).content
        except Exception:
            question = f"Could you please provide the missing details: {', '.join(missing)}?"
        return {"missing_fields": missing, "next_question": question}
        
    return {"missing_fields": [], "next_question": None}


# Node 3: Search vehicle deals (Mock scraping Kayak & Carhire.com.au)
def search_vehicle(state: CarHireState) -> Dict[str, Any]:
    location = state["location"]
    start_date = state["start_date"]
    end_date = state["end_date"]
    
    # Simple date parse to calculate days if format is correct
    days = 4
    try:
        d1 = datetime.strptime(start_date, "%Y-%m-%d")
        d2 = datetime.strptime(end_date, "%Y-%m-%d")
        days = max(1, (d2 - d1).days)
    except Exception:
        pass
        
    # Generate mock scraped data simulating search on kayak.com and carhire.com.au
    deals = [
        {
            "provider": "Kayak.com (via Hertz)",
            "location": location,
            "car_model": "Toyota Corolla (Sedan)",
            "price_per_day": 45.0,
            "total_price": 45.0 * days,
            "url": f"https://www.kayak.com.au/cars/{location}/{start_date}/{end_date}"
        },
        {
            "provider": "Carhire.com.au (via East Coast)",
            "location": location,
            "car_model": "Mitsubishi ASX (Compact SUV)",
            "price_per_day": 52.0,
            "total_price": 52.0 * days,
            "url": f"https://www.carhire.com.au/search?loc={location}&from={start_date}&to={end_date}"
        },
        {
            "provider": "Kayak.com (via Europcar)",
            "location": location,
            "car_model": "Hyundai i30 (Hatchback)",
            "price_per_day": 42.0,
            "total_price": 42.0 * days,
            "url": f"https://www.kayak.com.au/cars/{location}/{start_date}/{end_date}"
        }
    ]
    return {"scraped_deals": deals}


# Node 4: Synthesize response
def synthesize_response(state: CarHireState) -> Dict[str, Any]:
    if GEMINI_API_KEY:
        # Fallback if model unconfigured
        table = "| Provider | Car Model | Price/Day | Total Price | Link |\n|---|---|---|---|---|\n"
        for deal in state["scraped_deals"]:
            table += f"| {deal['provider']} | {deal['car_model']} | ${deal['price_per_day']} | ${deal['total_price']} | [Book]({deal['url']}) |\n"
        response = f"Here are the deals found for {state['location']} from {state['start_date']} to {state['end_date']}:\n\n{table}"
        return {"final_response": response}

    llm = ChatLiteLLM(model="gemini/gemini-1.5-flash", temperature=0.2)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a car hire comparison assistant. Based on the scraped vehicle deals, present the user with a formatted markdown summary/recommendation table including links to book. Highlight the best deals. Make sure the table looks professional."),
        ("user", "Deals: {deals}")
    ])
    chain = prompt_template | llm
    try:
        response = chain.invoke({"deals": str(state["scraped_deals"])}).content
    except Exception:
        # Fallback if model fails
        table = "| Provider | Car Model | Price/Day | Total Price | Link |\n|---|---|---|---|---|\n"
        for deal in state["scraped_deals"]:
            table += f"| {deal['provider']} | {deal['car_model']} | ${deal['price_per_day']} | ${deal['total_price']} | [Book]({deal['url']}) |\n"
        response = f"Here are the deals found for {state['location']} from {state['start_date']} to {state['end_date']}:\n\n{table}"
    return {"final_response": response}


# Conditional routing logic
def route_after_validation(state: CarHireState):
    if state.get("missing_fields"):
        return "ask_user"
    return "search"


# Construct the graph
workflow = StateGraph(CarHireState)

workflow.add_node("extract_parameters", extract_parameters)
workflow.add_node("validate_parameters", validate_parameters)
workflow.add_node("search_vehicle", search_vehicle)
workflow.add_node("synthesize_response", synthesize_response)

workflow.set_entry_point("extract_parameters")
workflow.add_edge("extract_parameters", "validate_parameters")

workflow.add_conditional_edges(
    "validate_parameters",
    route_after_validation,
    {
        "ask_user": END,
        "search": "search_vehicle"
    }
)

workflow.add_edge("search_vehicle", "synthesize_response")
workflow.add_edge("synthesize_response", END)

# Compile graph
car_hire_agent = workflow.compile()
