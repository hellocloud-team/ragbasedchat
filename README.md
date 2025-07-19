
(
  (
    100 - avg by(agent_hostname)(
      rate(node_cpu_seconds_total{agent_hostname=~"(?i:($hostname))", mode="idle"}[$interval])
    ) * 100
  ) < 90
)
and on(agent_hostname)
(
  avg by(agent_hostname)(
    (node_memory_MemTotal_bytes{agent_hostname=~"(?i:($hostname))"} - node_memory_MemAvailable_bytes{agent_hostname=~"(?i:($hostname))"})
    / node_memory_MemTotal_bytes{agent_hostname=~"(?i:($hostname))"} * 100
  ) < 90
)
and on(agent_hostname)
(
  avg by(agent_hostname)(
    (
      node_filesystem_size_bytes{agent_hostname=~"(?i:($hostname))", mountpoint="/"} 
      - node_filesystem_free_bytes{agent_hostname=~"(?i:($hostname))", mountpoint="/"}
    )
    / node_filesystem_size_bytes{agent_hostname=~"(?i:($hostname))", mountpoint="/"} * 100
  ) < 90
)
* 1





Chat with PDF using Langchain and Google Gemini

Used: 
1)python
2)Langchain
3)GoogleAI
4)Docker
5)Streamlit


Features

Streamlit Interface: Streamlit provides an easy-to-use interface for building interactive web applications in Python.
PDF Manipulation with PyPDF2: PyPDF2 is a Python library for working with PDF files, allowing users to manipulate PDF documents seamlessly.
Advanced Text Analysis with LangChain: LangChain is a library for advanced text analysis, including capabilities such as RecursiveCharacterTextSplitter for handling complex text structures effectively.
Google Generative AI Embeddings: Leveraging Google Generative AI for intelligent embeddings, which can enhance text analysis and understanding.
Vector Stores with FAISS: FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors, which can be useful for storing and retrieving embeddings efficiently.
Chat-based Generative AI: Integration of Google Generative AI for creating engaging conversational experiences within the application.
Question Answering Chain with LangChain: LangChain's question-answering capabilities enable the application to answer queries based on the content of PDF documents.
Prompt Templates with LangChain: Using predefined templates from the LangChain prompt library can streamline interactions and provide users with prompts for generating responses.
Dotenv Integration: Dotenv integration ensures that sensitive credentials and environment variables are securely managed within the application.
![image](https://github.com/hellocloud-team/ragbasedchat/assets/163302215/1732d9b7-803a-4f67-b0d1-18721100fd84)





Training Takeaways: Secure LLMs, Agentic AI & AI-Powered Java Applications

1. Building Secure LLMs (Large Language Models):

Data Security: Emphasize proper data sanitization and encryption throughout the pipeline—especially during data ingestion and storage.

Access Control: Implement strict access policies for LLM models, APIs, and inference endpoints.

Prompt Injection Defense: Use input validation, context separation, and guardrails to protect against prompt injection attacks.

Auditability: Ensure logging and monitoring are in place to track model usage, anomalies, and abuse patterns.

Ethical Safeguards: Integrate content filters, bias mitigation strategies, and explainability tools.


2. Developing Agentic AI Systems:

Autonomy with Oversight: Build agents that act autonomously but remain aligned with human objectives via continuous feedback loops.

Modular Architecture: Use composable tools like planners, memory modules, and tool interfaces for scalable design.

State Management: Implement persistent and contextual memory systems for long-term agent reasoning.

Security Controls: Prevent agents from taking harmful actions by enforcing constraints and validation at decision points.


3. AI-Powered Application Development with Java:

Framework Integration: Leverage Java ML libraries (e.g., Deep Java Library, Tribuo) or call Python models using JNI or REST APIs.

Microservices Architecture: Deploy AI functionalities as modular, secure microservices for scalability and maintainability.

Data Pipeline Handling: Use Java for robust ETL pipelines to prepare and manage input data for AI models.

Inference Management: Optimize model inference with caching, batching, and concurrency control in Java applications.

Monitoring & Logging: Use tools like Micrometer and Prometheus for observability of AI features in production.




