import base64
import json
import random
import string
import time
from typing import TypedDict, Optional, Dict, Any
import boto3
import httpx
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
import litellm

from app.app_exception import AppException
from app.llm_model import get_gemini_2_5_flash_model
from app.config import (
    GEMINI_API_KEY,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_REGION,
    AWS_S3_BUCKET,
)
from app.log import get_logger
from app.langfuse import get_langfuse_handler
from app.util import get_session_id
from langgraph.graph import StateGraph, START, END

log = get_logger("generate-image-agent")


class GenerateImageState(TypedDict):
    prompt: str
    safe_prompt: Optional[str]
    is_safe: Optional[bool]
    rejection_reason: Optional[str]
    image_bytes: Optional[bytes]
    image_url: Optional[str]
    final_response: Optional[str]


class SafetyCheckResult(BaseModel):
    is_safe: bool = Field(..., description="Whether the prompt is safe to generate")
    rejection_reason: Optional[str] = Field(None, description="Why the prompt was rejected if unsafe")
    safe_prompt: Optional[str] = Field(None, description="The enriched, safe prompt to use for image generation")


def screen_prompt(state: GenerateImageState) -> Dict[str, Any]:
    """
    Check if the user's prompt is safe to generate.
    Reject prompt if it contains adult, violent, harmful, or illegal content.
    If safe, enrich the prompt to generate a higher-quality image.
    """
    log.info("Screening prompt safety")
    llm = get_gemini_2_5_flash_model(temperature=0)
    structured_llm = llm.with_structured_output(SafetyCheckResult)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a strict safety screening assistant for an AI image generator.\n"
            "Analyze the user's prompt.\n"
            "You MUST reject the prompt if it contains, refers to, or requests:\n"
            "- Adult, sexually explicit, or pornographic content\n"
            "- Violence, blood, gore, self-harm, or cruelty\n"
            "- Hate speech, harassment, or direct attacks on individuals or groups\n"
            "- Illegal activities, weapons, drugs, or dangerous acts\n"
            "- Harming or exploitation of minors\n\n"
            "If the prompt is unsafe, you must reject it, set `is_safe` to false, and provide a polite, clear explanation in `rejection_reason`.\n"
            "If the prompt is safe:\n"
            "- Set `is_safe` to true.\n"
            "- Enrich/optimize the prompt for generating a beautiful, detailed image. Keep the user's core intent but add aesthetic details (lighting, style, resolution, clarity) that enhance the generation. Avoid any sensitive or unsafe additions. Write this in `safe_prompt`."
            "- Maximum image size is 1200px x 1200px. If user specify higher than that, update it to 1200px x 1200px."
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
                "langfuse_tags": ["generate_image_agent", "screen_prompt"]
            }
        } if handler else {}
        result = chain.invoke({"prompt": state["prompt"]}, config=config)
    except Exception as e:
        log.error(f"Error screening prompt: {str(e)}")
        raise AppException(f"failed to screen prompt safety: {str(e)}")

    if not result.is_safe:
        return {
            "is_safe": False,
            "rejection_reason": result.rejection_reason or "Unsafe prompt detected.",
            "safe_prompt": None
        }

    return {
        "is_safe": True,
        "rejection_reason": None,
        "safe_prompt": result.safe_prompt or state["prompt"]
    }


def generate_image(state: GenerateImageState) -> Dict[str, Any]:
    """
    Generate an image using Gemini Imagen 3 via LiteLLM
    """
    if not state.get("is_safe"):
        return {}

    log.info(f"Generating image with prompt: {state['safe_prompt']}")
    if not GEMINI_API_KEY:
        raise AppException("GEMINI_API_KEY is not configured in env")

    try:
        # LiteLLM call for gemini image generation (imagen-3.0-generate-002)
        response = litellm.image_generation(
            prompt=state["safe_prompt"],
            model="gemini/gemini-2.5-flash-image",
            api_key=GEMINI_API_KEY
        )

        image_data = response.data[0]
        if hasattr(image_data, "url") and image_data.url:
            img_response = httpx.get(image_data.url)
            img_response.raise_for_status()
            image_bytes = img_response.content
        elif hasattr(image_data, "b64_json") and image_data.b64_json:
            image_bytes = base64.b64decode(image_data.b64_json)
        elif isinstance(image_data, dict) and image_data.get("url"):
            img_response = httpx.get(image_data["url"])
            img_response.raise_for_status()
            image_bytes = img_response.content
        elif isinstance(image_data, dict) and image_data.get("b64_json"):
            image_bytes = base64.b64decode(image_data["b64_json"])
        else:
            raise AppException("Image generation response format is unrecognized")

        return {"image_bytes": image_bytes}
    except Exception as e:
        log.error(f"Error generating image: {str(e)}")
        raise AppException(f"Failed to generate image: {str(e)}")


def upload_to_s3(state: GenerateImageState) -> Dict[str, Any]:
    """
    Upload generated image bytes to AWS S3 and return the public URL
    """
    if not state.get("is_safe") or not state.get("image_bytes"):
        return {}

    log.info("Uploading image to AWS S3")
    if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, AWS_S3_BUCKET]):
        raise AppException("AWS S3 environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, AWS_S3_BUCKET) are not fully configured")

    timestamp = int(time.time() * 1000)
    rand_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    filename = f"{timestamp}-{rand_suffix}.png"

    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )

        s3_client.put_object(
            Bucket=AWS_S3_BUCKET,
            Key=filename,
            Body=state["image_bytes"],
            ContentType="image/png"
        )

        image_url = f"https://{AWS_S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{filename}"
        return {"image_url": image_url}
    except Exception as e:
        log.error(f"Error uploading to S3: {str(e)}")
        raise AppException(f"Failed to upload image to S3: {str(e)}")


def format_final_response(state: GenerateImageState) -> Dict[str, Any]:
    """
    Formulate the final JSON response
    """
    if not state.get("is_safe"):
        response_json = {
            "status": "rejected",
            "reason": state.get("rejection_reason"),
            "url": None
        }
    else:
        response_json = {
            "status": "success",
            "reason": None,
            "url": state.get("image_url")
        }

    return {"final_response": json.dumps(response_json, indent=4)}


def route_after_safety(state: GenerateImageState) -> str:
    if state.get("is_safe"):
        return "generate"
    return "format"


# Construct the StateGraph
workflow = StateGraph(GenerateImageState)

workflow.add_node("screen_prompt", screen_prompt)
workflow.add_node("generate_image", generate_image)
workflow.add_node("upload_to_s3", upload_to_s3)
workflow.add_node("format_final_response", format_final_response)

workflow.set_entry_point("screen_prompt")

workflow.add_conditional_edges(
    "screen_prompt",
    route_after_safety,
    {
        "generate": "generate_image",
        "format": "format_final_response"
    }
)

workflow.add_edge("generate_image", "upload_to_s3")
workflow.add_edge("upload_to_s3", "format_final_response")
workflow.add_edge("format_final_response", END)

generate_image_agent = workflow.compile()
