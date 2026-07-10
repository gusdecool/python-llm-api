from app.app_exception import AppException
from app.llm_model import get_gemini_2_5_flash_model
from typing import TypedDict, Optional, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from app.log import get_logger
from app.langfuse import get_langfuse_handler
from app.util import get_session_id

log = get_logger("marketing-agent")


class MarketingAgentState(TypedDict):
    prompt: str
    topic: Optional[str]
    drafts: Optional[List[Dict[str, Any]]]
    approved_option: Optional[int]
    final_response: Optional[str]
    next_question: Optional[str]


class DraftOption(BaseModel):
    id: int = Field(..., description="1-based index (1, 2, or 3)")
    headline: str = Field(..., description="Catchy headline for the marketing copy")
    body: str = Field(..., description="Persuasive body/ad copy")


class MarketingDrafts(BaseModel):
    topic: str = Field(..., description="The campaign topic or product name extracted from prompt")
    drafts: List[DraftOption] = Field(..., description="Exactly 3 diverse marketing copy options")


class DecisionDetails(BaseModel):
    action: str = Field(..., description="Either 'approve' or 'feedback'")
    approved_option: Optional[int] = Field(None, description="The approved option index (1, 2, or 3) if action is 'approve'")
    feedback: Optional[str] = Field(None, description="The feedback/edit request details if action is 'feedback'")


# Node 1: Generate or revise drafts
def generate_or_revise_drafts(state: MarketingAgentState) -> Dict[str, Any]:
    log.info("Running generate_or_revise_drafts")
    
    # If already approved, do nothing
    if state.get("approved_option") is not None:
        return {}

    llm = get_gemini_2_5_flash_model(temperature=0.7)
    structured_llm = llm.with_structured_output(MarketingDrafts)

    # Case A: Generating initial drafts
    if not state.get("drafts"):
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert copywriter. Create exactly 3 diverse marketing copy drafts based on the user's prompt.\n"
                "Provide a catchy headline and a persuasive body copy for each draft."
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
                    "langfuse_tags": ["marketing_agent", "initial_drafts"]
                }
            } if handler else {}
            res = chain.invoke({"prompt": state["prompt"]}, config=config)
            
            return {
                "topic": res.topic,
                "drafts": [d.model_dump() for d in res.drafts]
            }
        except Exception as e:
            log.error(f"Error in initial drafts generation: {e}")
            raise AppException(f"Failed to generate marketing drafts: {str(e)}")

    # Case B: Revising existing drafts based on feedback
    else:
        # Since we are here, there is feedback in state["prompt"]
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert copywriter. You have previously generated the following drafts for the topic '{topic}':\n"
                "{existing_drafts}\n\n"
                "The user provided feedback to revise these drafts:\n"
                "'{feedback}'\n\n"
                "Revise all 3 drafts according to the feedback. Maintain the same JSON schema with 3 drafts."
            ))
        ])
        
        drafts_str = "\n".join([f"Option {d['id']}: Headline: {d['headline']}\nBody: {d['body']}" for d in state["drafts"]])
        chain = prompt_template | structured_llm
        
        try:
            handler = get_langfuse_handler()
            config = {
                "callbacks": [handler],
                "metadata": {
                    "langfuse_session_id": get_session_id(),
                    "langfuse_tags": ["marketing_agent", "revise_drafts"]
                }
            } if handler else {}
            res = chain.invoke({
                "topic": state.get("topic") or "Campaign",
                "existing_drafts": drafts_str,
                "feedback": state["prompt"]
            }, config=config)
            
            return {
                "drafts": [d.model_dump() for d in res.drafts]
            }
        except Exception as e:
            log.error(f"Error in drafts revision: {e}")
            raise AppException(f"Failed to revise marketing drafts: {str(e)}")


# Node 2: Evaluate User Input (Approve or Feedback)
def evaluate_user_input(state: MarketingAgentState) -> Dict[str, Any]:
    log.info("Running evaluate_user_input")
    
    # We must determine if the latest prompt is an approval or feedback/revisions.
    # If it's the very first run (no drafts yet), we bypass this evaluation and go straight to asking for approval/feedback.
    if not state.get("drafts"):
        return {"approved_option": None, "next_question": None}

    llm = get_gemini_2_5_flash_model(temperature=0)
    structured_llm = llm.with_structured_output(DecisionDetails)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "Analyze the user prompt to determine if they are approving one of the marketing copy options (1, 2, or 3) "
            "or providing feedback/revisions.\n"
            "If they approve (e.g. 'Approve 1', 'I choose option 2', 'number 3 is good'), set action='approve' and approved_option to the selected number.\n"
            "If they want changes or ask for revisions, set action='feedback' and write their request in feedback."
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
                "langfuse_tags": ["marketing_agent", "evaluate_input"]
            }
        } if handler else {}
        decision = chain.invoke({"prompt": state["prompt"]}, config=config)
    except Exception as e:
        log.error(f"Error in evaluate_user_input: {e}")
        # Default fallback to feedback if we can't decide
        return {"approved_option": None, "next_question": "Could you please clarify if you want to approve an option or provide feedback?"}

    if decision.action == "approve" and decision.approved_option in (1, 2, 3):
        return {"approved_option": decision.approved_option, "next_question": None}
    
    # If feedback was detected, we clear approved_option and format the next question (which will be processed in draft revision)
    # Actually, we don't return next_question here because we want to trigger the node logic correctly.
    # If it is feedback, the flow will loop back or continue to request review.
    return {"approved_option": None, "next_question": None}


# Node 3: Finalize Campaign Assets
def finalize_campaign(state: MarketingAgentState) -> Dict[str, Any]:
    log.info("Running finalize_campaign")
    
    approved_idx = state.get("approved_option")
    if approved_idx is None:
         raise AppException("Cannot finalize campaign without approved option.")
         
    # Find the approved draft
    approved_draft = None
    for d in state.get("drafts", []):
        if d["id"] == approved_idx:
            approved_draft = d
            break
            
    if not approved_draft:
        # Fallback to option 1 if not found
        approved_draft = state["drafts"][0]
        
    llm = get_gemini_2_5_flash_model(temperature=0.5)
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a marketing strategist. Take the approved marketing copy option and compile a final campaign package.\n"
            "Include:\n"
            "1. The finalized Copy (Headline and Body).\n"
            "2. Recommended target demographics (age, interests, behaviors).\n"
            "3. A detailed prompt to generate campaign imagery or creative assets using an AI image generator (e.g. Midjourney/DALL-E).\n\n"
            "Format the output professionally in clean markdown."
        )),
        ("user", "Approved Copy Option:\nHeadline: {headline}\nBody: {body}")
    ])
    
    chain = prompt_template | llm
    try:
        handler = get_langfuse_handler()
        config = {
            "callbacks": [handler],
            "metadata": {
                "langfuse_session_id": get_session_id(),
                "langfuse_tags": ["marketing_agent", "finalize_campaign"]
            }
        } if handler else {}
        res = chain.invoke({
            "headline": approved_draft["headline"],
            "body": approved_draft["body"]
        }, config=config)
        
        return {"final_response": res.content, "next_question": None}
    except Exception as e:
        log.error(f"Error in finalize_campaign: {e}")
        raise AppException(f"Failed to finalize campaign: {str(e)}")


# Router/Condition function
def route_after_evaluation(state: MarketingAgentState) -> str:
    if state.get("approved_option") is not None:
        return "finalize"
    return "re_draft"


# Formatter node to set the interactive question
def format_next_question(state: MarketingAgentState) -> Dict[str, Any]:
    # Check if we already have final response, if so no next question
    if state.get("final_response"):
        return {"next_question": None}
        
    drafts_list = state.get("drafts") or []
    drafts_text = ""
    for d in drafts_list:
        drafts_text += f"\n**Option {d['id']}**\n* Headline: {d['headline']}\n* Body: {d['body']}\n"
        
    question = (
        f"Here are the marketing copy drafts I've created for the topic '{state.get('topic') or 'Campaign'}':\n"
        f"{drafts_text}\n"
        "Would you like to approve one of these options (e.g., 'Approve 1'), or do you have any feedback to revise them?"
    )
    return {"next_question": question}


# Build LangGraph workflow
builder = StateGraph(MarketingAgentState)

builder.add_node("evaluate", evaluate_user_input)
builder.add_node("draft", generate_or_revise_drafts)
builder.add_node("format_question", format_next_question)
builder.add_node("finalize", finalize_campaign)

# Graph edges
builder.add_edge(START, "evaluate")

# Route based on whether option was approved
builder.add_conditional_edges(
    "evaluate",
    route_after_evaluation,
    {
        "finalize": "finalize",
        "re_draft": "draft"
    }
)

builder.add_edge("draft", "format_question")
builder.add_edge("format_question", END)
builder.add_edge("finalize", END)

marketing_agent = builder.compile()
