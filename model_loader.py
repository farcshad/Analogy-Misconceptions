import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict


def load_model_huggingface(
    model_name: str,
    device: str = "cpu",
    load_in_8bit: bool = False,
    trust_remote_code: bool = False,
) -> ChatHuggingFace:
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

    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15,
    )

    chat_model = ChatHuggingFace(
        pipeline=HuggingFacePipeline(pipeline=hf_pipeline)
    )

    return chat_model