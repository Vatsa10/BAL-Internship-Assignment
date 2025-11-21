# # Financial Policy Analyst RAG (Asynchronous & Multi-Modal)

A high-performance Retrieval Augmented Generation (RAG) system designed for analyzing complex financial documents including IMF Article IV Reports and Central Bank Policy Papers. This tool leverages natural language processing and computer vision to extract, process, and generate insights from various document formats.

## Features

- **Multi-modal Document Processing**: Handles text, tables, and images within documents
- **Asynchronous Processing**: Efficiently processes large documents
- **Local-First Architecture**: Maintains document privacy with local processing (except for LLM calls)
- **Financial Document Specialization**: Optimized for financial and policy documents
- **Interactive Web Interface**: User-friendly interface for document management and queries

## Technical Stack

| Component       | Technology           | Description |
|----------------|----------------------|-------------|
| **UI**         | Gradio               | Web interface for document interaction |
| **PDF Engine** | PyMuPDF (fitz)       | High-performance PDF processing |
| **OCR**        | PaddleOCR            | Advanced OCR for tables and scanned content |
| **Vector DB**  | Qdrant               | High-performance vector database |
| **Embeddings** | FastEmbed (BGE-Small)| Efficient local text embeddings |
| **LLM**        | Gemini 2.0 Flash     | Advanced language model for analysis |

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Qdrant (local Docker or cloud instance)
- Google Gemini API Key (available from [Google AI Studio](https://ai.google.dev/))

### System Dependencies

For Linux/Ubuntu:
```bash
sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/financial-policy-rag.git
   cd financial-policy-rag
   ```

2. **Set up a virtual environment**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Unix or MacOS:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install packages individually:
   ```bash
   pip install pymupdf qdrant-client google-generativeai fastembed paddleocr opencv-python-headless pillow gradio requests python-dotenv
   ```

4. **Configure environment**
   Create a `.env` file in the project root with:
   ```
   GEMINI_API_KEY=your_gemini_api_key
   QDRANT_URL=http://localhost:6333  # For local Qdrant
   # QDRANT_API_KEY=your_cloud_key  # Only for Qdrant Cloud
   ```

## Running the Application

1. **Start Qdrant** (if using locally):
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

2. **Launch the application**:
   ```bash
   python main.py
   ```

3. **Access the interface**:
   - Open a web browser and go to `http://127.0.0.1:7860`
   - Upload PDFs or enter document URLs
   - Query the documents using natural language

## System Architecture

The application follows a modular architecture:

1. **Document Ingestion**: Processes various document types (PDF, scanned, digital)
2. **Content Analysis**:
   - Text extraction and chunking
   - Table recognition and conversion
   - Image/Chart analysis using computer vision
   - OCR for scanned documents
3. **Vector Database**: Stores document embeddings for efficient retrieval
4. **Query Processing**:
   - Converts queries to embeddings
   - Performs similarity search
   - Generates context-aware responses

## Supported Document Types

- PDF documents (text-based and scanned)
- Financial statements and reports
- Policy documents and white papers
- Research publications
- Presentation slides

## Troubleshooting

### Common Issues

1. **OCR Dependencies**
   - Linux: Install system dependencies as shown above
   - Windows: Install the latest Visual C++ Redistributable

2. **Qdrant Connection**
   - Ensure Docker is running for local Qdrant
   - Verify the Qdrant container is accessible at the specified URL

3. **API Limitations**
   - Monitor Gemini API usage to prevent rate limiting
   - Consider batching requests for large document sets

## Contributing

Contributions to improve the project are welcome. Please submit issues and pull requests through the project's GitHub repository.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For support or inquiries, please open an issue in the project repository.
