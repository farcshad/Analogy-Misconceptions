import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict, Optional, Dict, Any

DEFAULT_PIPELINE_KWARGS: Dict[str, Any] = {
    "temperature": 0.2,
    "max_new_tokens": 512,
}


def load_model_huggingface(
    model_name: str,
    device: str = "cpu",
    load_in_8bit: bool = False,
    trust_remote_code: bool = False,
    pipeline_kwargs: Optional[Dict[str, Any]] = None,
) -> ChatHuggingFace:
    """
    Load a causal LLM from Hugging Face and wrap it for use with LangGraph.

    Parameters:
    - model_name: model identifier on Hugging Face Hub
    - device: 'cpu' or other device string; controls `device_map` and dtype
    - load_in_8bit: whether to load model in 8-bit mode
    - trust_remote_code: whether to allow remote code execution for models
    - pipeline_kwargs: optional dict of keyword arguments passed to
      `transformers.pipeline(...)` so callers can customize generation params.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, trust_remote_code=trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device != "cpu" else None,
        load_in_8bit=load_in_8bit,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        trust_remote_code=trust_remote_code,
    )

    # Allow callers to supply any pipeline keyword arguments (e.g., temperature,
    # max_new_tokens, top_p, repetition_penalty) and merge them with the defaults.
    merged_pipeline_kwargs = dict(DEFAULT_PIPELINE_KWARGS)
    if pipeline_kwargs:
        merged_pipeline_kwargs.update(pipeline_kwargs)
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        **merged_pipeline_kwargs,
    )

    chat_model = ChatHuggingFace(
        pipeline=HuggingFacePipeline(pipeline=hf_pipeline)
    )

    return chat_model
