from app.app_exception import AppException
from app.log import get_logger
import sys
from app.agents import choose_agent, car_hire_agent, weather_agent, generate_image_agent
from sqlmodel import Session
from app.db import engine, init_db

log = get_logger("cli-main")


def run_generate_image_flow(initial_prompt: str, session: Session = None) -> None:
    state = {
        "prompt": initial_prompt,
        "safe_prompt": None,
        "is_safe": None,
        "rejection_reason": None,
        "image_bytes": None,
        "image_url": None,
        "final_response": None
    }
    config = {
        "configurable": {
            "session": session,
            "user_id": "default_user"
        }
    } if session else {}
    result = generate_image_agent.invoke(state, config=config)
    print(f"\nAgent: {result.get('final_response') or 'Image generation complete.'}")


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


def run_weather_flow(initial_prompt: str, session: Session = None) -> None:
    state = {
        "prompt": initial_prompt,
        "location": None,
        "date": None,
        "missing_fields": [],
        "next_question": None,
        "weather_data": None,
        "final_response": None
    }
    config = {
        "configurable": {
            "session": session,
            "user_id": "default_user"
        }
    } if session else {}
    
    result = weather_agent.invoke(state, config=config)
    
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
        result = weather_agent.invoke(state, config=config)
        
    print(f"\nAgent: {result.get('final_response') or 'Search complete.'}")


def main() -> None:
    init_db()
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

        with Session(engine) as session:
            choice = choose_agent(prompt, session=session, user_id="default_user")

            try :
                if choice.action == "direct_answer":
                    log.info("agent can direct answer")
                    print(f"\nAgent: {choice.direct_response}")
                elif choice.action == "weather_agent":
                    print("\n[Routing to Weather Agent...]")
                    run_weather_flow(prompt, session=session)
                elif choice.action == "car_hire_agent":
                    print("\n[Routing to Car Hire Agent...]")
                    run_car_hire_flow(prompt)
                elif choice.action == "generate_image_agent":
                    print("\n[Routing to Generate Image Agent...]")
                    run_generate_image_flow(prompt, session=session)
                elif choice.action == "unsupported":
                    print(f"\nAgent: {choice.direct_response}")
                else:
                    print("\nAgent: I cannot do that at the moment.")
            except AppException as e:
                # print the error message and ask user input again
                print("\nAgent: " + e.message)
                # let the process continue to ask for another input
                continue


if __name__ == "__main__":
    main()

