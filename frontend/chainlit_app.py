import chainlit as cl
import requests
from datetime import datetime
import base64
import json
import asyncio

# API endpoint
API_BASE_URL = "http://localhost:8001"

async def process_query(query: str, user_id: str):
    """Send query to FastAPI backend and process response"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={
                "query": query,
                "data_path": "data",
                "user_id": user_id
            }
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error communicating with API: {str(e)}")

@cl.on_chat_start
async def start():
    user_id = f"user_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    cl.user_session.set("user_id", user_id)
    current_chunk = ""
    msg =cl.Message(content="")
    token_list = "Welcome! I'm ready to answer questions about your documents."
    for token in token_list.split(' '):
        current_chunk = token + " "
        await asyncio.sleep(0.15)
        await msg.stream_token(current_chunk)
    await msg.send()

@cl.on_message
async def main(message: cl.Message):
    user_id = cl.user_session.get("user_id")
    msg1 = cl.Message(content="")
    token_list = "Thinking... ... ... .. .. . . . ."
    for token in token_list.split(' '):
        current_chunk = token + " "
        await asyncio.sleep(0.3)
        await msg1.stream_token(current_chunk)
    await msg1.send()
    
    try:
        response_data = await process_query(message.content, user_id)
        # print("response_data", response_data)
        response_dict = response_data["response"]
        print("response_dict", response_dict)
        elements = []
        total_text = []
        if "source_nodes" in response_data:
            for idx, node in enumerate(response_data["source_nodes"]):
                pdf_name = node['metadata'].get('pdf_name', 'Unknown')
                page_num = node['metadata'].get('page_num', 'Unknown')
                source_id = f"Source: {pdf_name}, Page: {page_num}"
                total_text.append(node['text'])
                if "image_base64" in node:
                    elements.append(
                        cl.Image(
                            name=f"source_image_{len(elements)}",
                            display="inline",
                            content=base64.b64decode(node["image_base64"]),
                            caption=f"Source: {node['metadata']['pdf_name']}, Page: {node['metadata']['page_num']}"
                        ))

        actions=[cl.Action(name="show_source",
                            payload={"source_id": source_id, "text": total_text},
                            label="Click to view source")]
        await cl.Message(
            content=response_dict,
            elements=elements,
            actions=actions
        ).send()

    except Exception as e:
        await cl.Message(
            content=f"Error: {str(e)}",
            type="error"
        ).send()

@cl.action_callback("show_source")
async def on_action(action):
    """Handle clicks to show source content"""
    try:
        # Extract values from the action value
        source_id = action.payload.get("source_id", "Unknown Source")
        text = action.payload.get("text", "No content available")
        
        await cl.Message(
            content=f"📄 {source_id}\n\n{text}",
            author="Source"
        ).send()
    except Exception as e:
        await cl.Message(
            content=f"Error displaying source: {str(e)}",
            type="error"
        ).send()