from app.log import get_logger
import sys
from app.agents import choose_agent, car_hire_agent, weather_agent

log = get_logger("cli-main")

def run_car_hire_flow(initial_prompt: str) -> None:
    state = {
        "prompt": initial_prompt,
        "location": None,
        "start_date": None,
        "end_date": None,
        "missing_fields": [],
        "next_question": None,
        "scraped_deals": None,
        "final_response": None
    }
    
    result = car_hire_agent.invoke(state)
    
    while result.get("next_question"):
        print(f"\nAgent: {result['next_question']}")
        try:
            answer = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            sys.exit(0)
            
        if not answer:
            continue
            
        state = {
            "prompt": answer,
            "location": result.get("location"),
            "start_date": result.get("start_date"),
            "end_date": result.get("end_date"),
            "missing_fields": [],
            "next_question": None,
            "scraped_deals": None,
            "final_response": None
        }
        result = car_hire_agent.invoke(state)
        
    print(f"\nAgent: {result.get('final_response') or 'Search complete.'}")


def run_weather_flow(initial_prompt: str) -> None:
    state = {
        "prompt": initial_prompt,
        "location": None,
        "date": None,
        "missing_fields": [],
        "next_question": None,
        "weather_data": None,
        "final_response": None
    }
    
    result = weather_agent.invoke(state)
    
    while result.get("next_question"):
        print(f"\nAgent: {result['next_question']}")
        try:
            answer = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            sys.exit(0)
            
        if not answer:
            continue
            
        state = {
            "prompt": answer,
            "location": result.get("location"),
            "date": result.get("date"),
            "missing_fields": [],
            "next_question": None,
            "weather_data": None,
            "final_response": None
        }
        result = weather_agent.invoke(state)
        
    print(f"\nAgent: {result.get('final_response') or 'Search complete.'}")


def main() -> None:
    print("==================================================")
    print("Welcome to the LLM AI Agent interactive CLI!")
    print("Type your query or type 'exit' / 'quit' to close.")
    print("==================================================")
    
    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
            
        if not prompt:
            continue
            
        if prompt.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
            
        choice = choose_agent(prompt)
        
        if choice.action == "direct_answer":
            log.info("agent can direct answer")
            print(f"\nAgent: {choice.direct_response}")
        elif choice.action == "weather_agent":
            print("\n[Routing to Weather Agent...]")
            run_weather_flow(prompt)
        elif choice.action == "car_hire_agent":
            print("\n[Routing to Car Hire Agent...]")
            run_car_hire_flow(prompt)
        elif choice.action == "unsupported":
            print(f"\nAgent: {choice.direct_response}")
        else:
            print("\nAgent: I cannot do that at the moment.")


if __name__ == "__main__":
    main()
