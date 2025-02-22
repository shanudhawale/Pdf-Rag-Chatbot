import logging
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from llama_index.core import StorageContext, VectorStoreIndex, ServiceContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import os
from datetime import datetime
from pathlib import Path
import re
from copy import deepcopy
from llama_index.core.query_engine import CustomQueryEngine, SimpleMultiModalQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.schema import ImageNode, NodeWithScore, MetadataMode, TextNode, ImageDocument
from llama_index.core.prompts import PromptTemplate
from llama_index.core.base.response.schema import Response
from dotenv import load_dotenv
from InstructorEmbedding import INSTRUCTOR
from llama_index.core import Settings
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding
import base64
import hashlib
import shutil
from llama_index.core import Document
from llama_index.core.memory import ChatMemoryBuffer
import json

class InstructorEmbeddings(BaseEmbedding):
    _model: INSTRUCTOR = PrivateAttr()
    _instruction: str = PrivateAttr()

    def __init__(
        self,
        instructor_model_name: str = "hkunlp/instructor-base",
        instruction: str = "Represent a Financial document with Titles, Paragraphs, and Images for semantic search as it can contain financial diagrams and tables:",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._model = INSTRUCTOR(instructor_model_name)
        self._instruction = instruction

    @classmethod
    def class_name(cls) -> str:
        return "instructor"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        embeddings = self._model.encode([[self._instruction, query]])
        return embeddings[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        embeddings = self._model.encode([[self._instruction, text]])
        return embeddings[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(
            [[self._instruction, text] for text in texts]
        )
        return embeddings


embed_model = InstructorEmbeddings(embed_batch_size=2)

Settings.embed_model = embed_model
Settings.chunk_size = 512

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="RAG API")

CHROMA_DB_PATH = "./chroma_db2"
db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection_name = "quickstart"
chroma_collection = db.get_or_create_collection(collection_name)

# Add this with your other global variables
current_index: Optional[VectorStoreIndex] = None

def initialize_index():
    """Initialize the index from ChromaDB"""
    global current_index
    
    try:
        # Create vector store and storage context
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Initialize or load the index
        if chroma_collection.count() > 0:
            logger.info("Loading existing ChromaDB index...")
            current_index = VectorStoreIndex.from_vector_store(
                vector_store,
                embed_model=embed_model
            )
        else:
            logger.warning("No documents found in ChromaDB")
            current_index = None
            
        return current_index
    
    except Exception as e:
        logger.error(f"Error initializing index: {e}")
        raise

# Initialize index when app starts
@app.on_event("startup")
async def startup_event():
    """Initialize the index when the app starts"""
    initialize_index()

class QueryRequest(BaseModel):   
    query: str = Field(..., description="User query")
    data_path: Optional[str] = Field(default="data", description="Path to document directory")
    user_id: str = Field(..., description="Unique user identifier")

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: datetime

# Initialize global variables
index_cache: Dict[str, VectorStoreIndex] = {}
chat_history: Dict[str, List[ChatMessage]] = {}


QA_PROMPT_TMPL = """\
Below we give parsed text from slides in two different formats, as well as the image.

We parse the text in both 'markdown' mode as well as 'raw text' mode. Markdown mode attempts \
to convert relevant diagrams into tables, whereas raw text tries to maintain the rough spatial \
layout of the text.

Chat History:
{chat_history}

Use the image information first and foremost. ONLY use the text/markdown information 
if you can't understand the image.

---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query. Explain whether you got the answer
from the parsed markdown or raw text or image, and if there's discrepancies, and your reasoning for the final answer.

Query: {query_str}
Please provide a detailed response :
1. Directly answers the question
2. References specific parts of the documents (line numbers and page numbers) in the context information in one line.Describes relevant images when they support the answer
3. Maintains continuity with previous conversation
Answer: 

Given below is XML that describes the information to extract from this document and the tags to extract it into.


<output>
    <list name="result_respnse" description="Bullet points regarding the query">
        <object>
            <string name="explanation"/>
            <string name="explanation2"/>
            <string name="explanation3"/>
        </object>
    </list>
    <list name="refrence" description="multiplequery reference">
        <object>
            <string name="page_number, pdf_name"/>
        </object>
    </list>
</output>



ONLY return a valid JSON object (no other text is necessary). The JSON MUST conform to the XML format, including any types and format requests e.g. requests for lists, objects and specific types. Be correct and concise.
"""


CONTEXT_PROMPT = PromptTemplate(QA_PROMPT_TMPL)

class MultiModalConversationalEngine(CustomQueryEngine):
    """Custom query engine for multimodal conversational RAG"""
    
    def intialize_engine(
        self,
        retriever,
        multi_modal_llm: OpenAIMultiModal,
        memory_buffer: ChatMemoryBuffer = None,
        context_prompt: PromptTemplate = CONTEXT_PROMPT,
    ):
        self._retriever = retriever
        self._llm = multi_modal_llm
        self._memory = memory_buffer or ChatMemoryBuffer.from_defaults(token_limit=2)
        self._context_prompt = context_prompt


    def _create_image_documents(self, image_paths):
        """Create image documents for the OpenAI multimodal model"""
        image_documents = []
        #print("@@@@@@@@@",image_paths)
        for path in image_paths:
            try:
                logger.info(f'@@@@@@@{path} {type(path)}')
                path = path.replace("\\","/")
                #logger.info(f'@@@@@@@{path}')
                if path and os.path.exists(path):  # Check if path exists
                    with open(path, "rb") as f:
                        image_data = f.read()
                        image_doc = ImageDocument(
                            image_data=image_data,
                            image_path=path,
                        )
                        image_documents.append(image_doc)
               	else:
                    logger.warning(f"Image path does not exist: {path}")
            except Exception as e:
                logger.error(f"Error reading image {path}: {e}")
        return image_documents

    def process_response(self, response_text: str) -> Dict[str, Any]:
        """
        Process and restructure the response to combine explanations
        """
        try:
            cleaned_text = response_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()
            response_data = json.loads(cleaned_text)
            
            if "result_respnse" in response_data and isinstance(response_data["result_respnse"], list):
                if len(response_data["result_respnse"]) > 0:
                    explanations = response_data["result_respnse"][0]
                    combined_explanation = " ".join(filter(None, [
                        explanations.get("explanation", ""),
                        explanations.get("explanation2", ""),
                        explanations.get("explanation3", "")
                    ]))
                else:
                    combined_explanation = "No explanation provided."
                
                processed_response = {
                    "result_response": {
                        "explanation": combined_explanation.strip()
                    },
                    "refrence": []
                }
                
                if "refrence" in response_data:
                    for ref in response_data["refrence"]:
                        if isinstance(ref, dict) and "page_number, pdf_name" in ref:
                            try:
                                page_num, pdf_name = ref["page_number, pdf_name"].split(", ")
                                processed_response["refrence"].append({
                                    "page_number": page_num.strip(),
                                    "pdf_name": pdf_name.strip()
                                })
                            except ValueError:
                                logger.warning(f"Invalid reference format: {ref}")
                
                source_links = []
                for ref in processed_response["refrence"]:
                    source_link = f"[Source: {ref['pdf_name']}, Page: {ref['page_number']}]"
                    source_links.append(source_link)

                if source_links:
                    combined_explanation += "\n\nSources:\n" + "\n".join(source_links)
                
                processed_response = {
                    "result_response": {
                        "explanation": combined_explanation.strip()
                    },
                    "refrence": processed_response["refrence"],
                    "source_links": source_links
                }
                
                return processed_response
            
            return response_data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON Decode Error: {str(e)}\nResponse text: {response_text}")
            return {
                "result_response": {
                    "explanation": str(response_text)
                },
                 "refrence": [],
                 "source_links": []
            }

    def custom_query(self, query_str: str) -> Response:
        """Process query with context and chat history"""
        

        # Get chat history
        chat_history = self._memory.get()
        chat_history_str = "\n".join([
            f"{msg.role}: {msg.content}"
            for msg in chat_history
        ])
        chat_history_str_content = "\n".join([
            f"{msg.content}"
            for msg in chat_history
        ])
        # Get relevant documents
        retrieved_nodes = self._retriever.retrieve(query_str+"\n"+chat_history_str_content)
        print("@##@",retrieved_nodes)
        # Prepare context from nodes
        context_chunks = []
        image_nodes = []
        
        for node in retrieved_nodes:
            # Handle text content
            if isinstance(node, NodeWithScore):
                print(node.node.metadata)
                text = node.node.get_content(metadata_mode=MetadataMode.ALL)
                metadata = node.node.metadata
                source_info = f"\nSource: {metadata.get('pdf_name', 'Unknown')}, Page: {metadata.get('page_num', 'Unknown')}"
                context_chunks.append(text + source_info)
                
                # Check for images in metadata
                if "image_path" in node.node.metadata:
                    image_nodes.append(node)
            
            # Handle image nodes
            if isinstance(node.node, ImageNode):
                image_nodes.append(node)
        # Combine text context
        context_text = "\n\n".join(context_chunks)
        
        #chat_history = self._memory.get()
        #print("!@#!!", chat_history)
        #chat_history_str = "\n".join([
        #    f"{msg.role}: {msg.content}"
        #    for msg in chat_history
        #])
        
        # Format prompt with context and history
        prompt = self._context_prompt.format(
            chat_history=chat_history_str,
            context_str=context_text,
            query_str=query_str
        )
        
        # Collect all image paths
        all_image_paths = []
        for node in image_nodes:
            if isinstance(node.node, ImageNode):
                if hasattr(node.node, 'image_path'):
                    all_image_paths.append(node.node.image_path)
            elif "image_path" in node.node.metadata:
                paths = node.node.metadata["image_path"]
                if isinstance(paths, list):
                    all_image_paths.extend(paths)
                else:
                    all_image_paths.append(paths)
        
        # Remove duplicates while preserving order
        all_image_paths = list(dict.fromkeys(all_image_paths))
        
        # Create image documents
        image_documents = self._create_image_documents(all_image_paths)
        print(image_documents)
        if not image_documents:
            logger.warning("No valid images found in the retrieved nodes")
        
        # Add text response with images
        text_response = self._llm.complete(
            prompt=prompt,
            image_documents=image_documents
        )
        
        final_response = str(text_response)
        print("Final response::",final_response)
        # Process and restructure the response
        processed_response = self.process_response(final_response)
        final_response_str = str(processed_response["result_response"]["explanation"])
        print(">>>>>>>>>",processed_response["refrence"])
        page_number_retrived = [str(processed_response["refrence"][i]["page_number"].split(' ')[-1]) for i in range(len(processed_response["refrence"]))]
        pdf_name_retrived = [str(processed_response["refrence"][i]["pdf_name"]) for i in range(len(processed_response["refrence"]))]
        print(page_number_retrived, pdf_name_retrived)
        #print(final_response_str)
        
        self._memory.put(ChatMessage(role="user", content=f"{query_str}", timestamp=datetime.now()))
        self._memory.put(ChatMessage(role="assistant",content=f"{final_response_str}", timestamp=datetime.now()))
        
        # Process source nodes and include base64 images
        seen_nodes = set()
        source_nodes_with_images = []
        print("@@@@@",retrieved_nodes)
        for node in retrieved_nodes:
            if isinstance(node, NodeWithScore):
                # Get the unique identifier tuple
                pdf_name = node.node.metadata.get("pdf_name", "")
                page_num = node.node.metadata.get("page_num", 0)
                node_id = (pdf_name, page_num)
                
                if node_id not in seen_nodes and str(page_num) in page_number_retrived and str(pdf_name) in pdf_name_retrived:
                    seen_nodes.add(node_id)
                    node_data = {
                        "text": node.node.get_content(),
                        "metadata": node.node.metadata,
                        "source_link": f"[Source: {pdf_name}, Page: {page_num}]",
                    }
                    if "image_path" in node.node.metadata:
                        node_data["image_base64"] = get_base64_image(node.node.metadata['image_path'])
                    
                    source_nodes_with_images.append(node_data)
        
        return Response(final_response_str, source_nodes=source_nodes_with_images)


def get_base64_image(image_path: str) -> str:
    try:
        image_path = image_path.replace("\\", "/") 
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error reading image {image_path}: {e}")
        return ""

@app.post("/query")
async def query_documents(request: QueryRequest):
    try:
        if current_index is None:
            raise HTTPException(
                status_code=400,
                detail="No indexed documents found. Please index documents first."
            )
        
        # Initialize multimodal LLM
        gpt_4v = OpenAIMultiModal(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            max_new_tokens=4096
        )
        
        # Initialize memory buffer
        memory_key = f"memory_{request.user_id}"
        if memory_key not in chat_history:
            chat_history[memory_key] = ChatMemoryBuffer.from_defaults()
        
        # Create custom query engine
        query_engine = MultiModalConversationalEngine()
        query_engine.intialize_engine(
            retriever=current_index.as_retriever(similarity_top_k=3),
            multi_modal_llm=gpt_4v,
            memory_buffer=chat_history[memory_key]
        )
        
        # Get response
        response = query_engine.custom_query(request.query)
        
        return {
            "response": str(response),
            "source_nodes": response.source_nodes
        }
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat_history/{user_id}")
async def get_chat_history(user_id: str):
    if user_id not in chat_history:
        return {"messages": []}
    return {"messages": chat_history[user_id]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
