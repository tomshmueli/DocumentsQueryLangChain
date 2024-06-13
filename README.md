
# Documents analyzer using Lanchain

This project demonstrates how to set up and use a Retrieval-Augmented Generation (RAG) system with Langchain.

## Prerequisites

### Windows Users
1. Follow the guide to install the Microsoft C++ Build Tools. Ensure you follow through to the last step to set the environment variable path.

### MacOS Users
1. Install the ONNX runtime dependency for ChromaDB using Conda:
   ```bash
   conda install onnxruntime -c conda-forge
   ```

### General Prerequisites
1. Ensure you have Python3 installed. It's recommended to use a virtual environment for this project.

## Setup

1. **Extract files** 

2. **Set up a virtual environment**:
   - On Windows:
     ```bash
     python -m venv venv
     .\venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project directory with the following content:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

## Running the Project

#### Option 1
   **Run the main script**:
   ```bash
   python main.py
   ```
   this will run the whole program

#### Option 2 (troubleshooting)
1. **Create the Chroma Database**:
   ```bash
   python create_database.py
   ```

2. **Query the Chroma Database** (replace `{a question}` with your actual query):
   ```bash
   python query_data.py "what is the title of the document?"
   ```

3. **Run the main script**:
   ```bash
   python main.py
   ```

## Additional Resources Used

- [Langchain Documentation](https://langchain.com/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [OpenAI API Documentation](https://beta.openai.com/docs/)
