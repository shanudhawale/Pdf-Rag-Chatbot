# PDF RAG Chatbot

A sophisticated PDF chatbot leveraging Retrieval-Augmented Generation (RAG) with multimodal capabilities, enabling intelligent conversations about both textual and visual content from PDF documents.

## Features

- **Advanced RAG Implementation**: Custom multimodal RAG system using LlamaIndex + LlamaParse + Instructor Embeddings + ChromaDB + GPT-4V + Guardrails via Custom Prompt
- **PDF Processing**: Intelligent handling of PDFs including text and images
- **Interactive Chat Interface**: User-friendly Chainlit interface with streaming responses
- **Source Attribution**: tracking of source documents and page numbers
- **Visual Context**: Dynamic display of relevant images from PDFs
- **Conversation Memory**: Context-aware responses using chat history

## Technical Architecture

### RAG Implementation Details

The system implements a sophisticated RAG architecture with the following components:

1. **Document Processing**
   - Processes PDFs to extract both text as markdowns and spatial text using LlamaParse
   - Processes PDFs to extract images using LlamaParse (can also used pymupdf)
   - Maintains spatial relationships and document structure
   - Creates separate nodes for text and image content and create a metadata rich document

2. **Embedding System**
   - Uses Instructor Embeddings from Hugging Face "hkunlp/instructor-base" for semantic understanding
   - Custom instruction tuning for financial document context by using a custom prompt specifying the context of the document
   - Batch processing capability for efficient embedding generation

3. **Vector Storage**
   - ChromaDB as the vector store and the indexes are created using the Instructor Embeddings
   - Persistent storage for document embeddings
   - Efficient similarity search capabilities

4. **Retrieval System**
   - Top-k similarity search for relevant content using ChromaDB indexes
   - Hybrid retrieval combining text and image nodes
   - Context-aware document fetching and passing the context to the LLM for response generation via guardrails included in the QA Custom Prompt

### Multimodal Conversational Query Engine

The `MultiModalConversationalEngine` is a custom implementation that handles complex multimodal interactions:

1. **Query Processing**
   ```python
   def custom_query(self, query_str: str):
       # Retrieves relevant documents
       # Processes both text and image nodes
       # Maintains conversation context
       # Returns structured response with sources and images as base64 encoded strings
   ```
2. **Image Processing**
   - Base64 encoding for frontend display
   - Maintains image metadata and relationships

3. **Context Management**
   - Chat history integration
   - Source document tracking
   - Response formatting and structuring

4. **Response Generation**
   - GPT-4V for multimodal understanding
   - Structured JSON responses via guardrails included in the QA Custom Prompt
   - Source attribution and reference tracking

## 🛠️ Technology Stack

### Backend Components
- FastAPI for API endpoints
- LlamaIndex for RAG implementation
- ChromaDB for vector storage
- Instructor Embeddings
- GPT-4V for multimodal processing

### Frontend Components
- Chainlit for interactive interface
- Dynamic base64 encoded image rendering

## 🚀 Installation

### Local Setup

1. Create a Virtual Environment and activate it:

```bash
python -m venv venv
source venv/bin/activate
```
2. Install dependencies:

```bash
pip install -r Pdf-Rag-Chatbot/requirements-new.txt
``` 
3. Clone the repository:

```bash
git clone https://github.com/yourusername/pdf-rag-chatbot.git
cd pdf-rag-chatbot
```
4. Create a `.env` file and add the environment variables:

```bash
LLAMA_API_KEY=<your_api_key_here>
OPENAI_API_KEY=<your_api_key_here>
```
6. Run the application frontend:

```bash
chainlit run app.py --port 8000 --host 0.0.0.0
```
7. Run the application backend:

```bash
uvicorn appv2:app --reload --port 8001 --host 0.0.0.0
```
8. Reverse Proxy using ngrok with static domain:

```bash
ngrok http --url=<url_value> http://0.0.0.0:8000
```
## Layout Demo

![Layout Demo](demo.gif)

## 🔧 Troubleshooting

Common issues and solutions:
- API key configuration
- Memory management
- Image processing errors
- Vector store optimization