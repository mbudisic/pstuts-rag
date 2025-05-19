# Task 1: Defining your Problem and Audience

**Problem:** Locating specific Photoshop information in long video tutorial transcripts is difficult and time-consuming.

**Users and their Problem:** Photoshop learners (designers, photographers, students, hobbyists) often struggle with inefficiently searching video tutorials for specific techniques. They need a quick way to query tutorial content for direct, concise answers, saving time and reducing learning frustration.

# Task 2: Propose a Solution

**Our Solution:** An agentic Retrieval Augmented Generation (RAG) system answers Adobe Photoshop questions. Users interact via a chat interface (Chainlit, as seen in `app.py`). The system queries its tutorial transcript knowledge base and can use Tavily for web searches, providing comprehensive answers.

**The Tech Stack 🛠️:** (Primary sources: `app.py`, `pstuts_rag/datastore.py`, `pyproject.toml`, `README.md`)

*   **LLM:** OpenAI model (e.g., `gpt-4.1-mini` in `app.py`), selected for strong language capabilities.
*   **Embedding Model:** An open-source model, `Snowflake/snowflake-arctic-embed-s` (see `Fine_Tuning_Embedding_for_PSTuts.ipynb`), fine-tuned for domain-specific relevance.
*   **Orchestration:** LangChain & LangGraph (`app.py`), for building the RAG application and managing agent workflows.
*   **Vector Database:** Qdrant (`pstuts_rag/datastore.py`), for efficient semantic search of tutorial transcripts.
*   **Monitoring:** W&B (Weights & Biases) is present in `notebooks/` and `Fine_Tuning_Embedding_for_PSTuts.ipynb`, used for experiment tracking during development.
*   **Evaluation:** RAGAS (`evaluate_rag.ipynb`, `pyproject.toml`), for assessing RAG pipeline quality.
*   **User Interface:** Chainlit (`app.py`, `chainlit.md`), for creating the interactive chat application.
*   **Serving & Inference:** Docker (`Dockerfile`), for containerized deployment (e.g., on Hugging Face Spaces, as suggested in `README.md` metadata).

**The Role of Agents 🕵️‍♂️:** (Primary source: `app.py`)

The system uses a LangGraph-orchestrated multi-agent approach:
1.  **Supervisor Agent:** Manages the overall workflow. It receives the user query and routes it to the appropriate specialized agent based on its interpretation of the query (defined in `SUPERVISOR_SYSTEM` prompt and `create_team_supervisor` in `app.py`).
2.  **Video Archive Agent (`VIDEOARCHIVE`):** This is the RAG agent. It queries the Qdrant vector store of Photoshop tutorial transcripts to find relevant information and generates an answer based on this retrieved context. (Uses `create_rag_node` from `pstuts_rag.agent_rag`).
3.  **Adobe Help Agent (`ADOBEHELP`):** This agent uses the Tavily API to perform web searches if the supervisor deems it necessary for broader or more current information. (Uses `create_tavily_node` from `pstuts_rag.agent_tavily`).
The supervisor then determines if the answer is complete or if further steps are needed.

# Task 3: Dealing with the Data

Our Photoshop RAG system uses specific data and chunking for accurate, relevant answers.

**1. Data Sources & External APIs 📊+🌐:**

*   **Primary Data Source:** JSON transcript files from Photoshop video tutorials (e.g., `data/dev.json`, loaded in `app.py`). *Purpose:* Core knowledge base, processed and indexed in Qdrant for semantic search.
*   **External API:** Tavily Search API (configured in `app.py`). *Purpose:* Augments knowledge with web search results via the `ADOBEHELP` agent for current or broader topics.

**2. Default Chunking Strategy 🧠✂️:** (Source: `pstuts_rag/datastore.py`'s `chunk_transcripts` function)

A **semantic chunking** strategy is employed:
1.  **Initial Loading:** Transcripts are loaded both entirely per video (`VideoTranscriptBulkLoader`) and as individual sentences/segments with timestamps (`VideoTranscriptChunkLoader`).
2.  **Semantic Splitting:** `SemanticChunker` (LangChain, using `OpenAIEmbeddings`) splits full transcripts into semantically coherent chunks.
3.  **Metadata Enrichment:** These semantic chunks are enriched with start/end times by mapping them back to the original timestamped sentences.

*   **Why this Strategy?** Ensures topically focused chunks for better retrieval relevance, provides richer context to the LLM, and allows linking back to video timestamps.

**3. [Optional] Specific Data Needs for Other Parts 🧩:**

*   **Embedding Model Fine-Tuning (Task 6):** The `Fine_Tuning_Embedding_for_PSTuts.ipynb` notebook generated/used a question-passage dataset from Photoshop tutorials (detailed in `dataset_card.md`) to adapt the `Snowflake/snowflake-arctic-embed-s` model for better Photoshop-specific retrieval.
*   **Evaluation & Golden Dataset (Tasks 5 & 7):** The process for generating the "Golden Data Set" (question-context-answer triplets) used for RAGAS evaluation is detailed in the `create_golden_dataset.ipynb` notebook within the `PsTuts-VQA-Data-Operations` repository ([https://github.com/mbudisic/PsTuts-VQA-Data-Operations](https://github.com/mbudisic/PsTuts-VQA-Data-Operations)). This dataset, subsequently referred to as `golden_small_hf` on Hugging Face, was then used in the main project's `evaluate_rag.ipynb` for benchmarking.

# Task 4: Building a Quick End-to-End Prototype

An end-to-end prototype RAG system for Photoshop tutorials is built and deployable.

**1. The Prototype Application 🖥️:** (Source: `app.py`)

The `app.py` script is the core prototype. It uses Chainlit for the UI, LangChain/LangGraph for orchestration, Qdrant for the vector store, and OpenAI models for embeddings and generation. It loads data, builds the RAG chain, and manages the agentic workflow for user queries.

**2. Deployment 🚀 (Hugging Face Space):**

The repository is structured for Hugging Face Space deployment:
*   `README.md` contains Hugging Face Space metadata (e.g., `sdk: docker`).
*   A `Dockerfile` enables containerization for deployment.
This setup indicates the prototype is packaged for public deployment.

# Task 5: Creating a Golden Test Data Set

The creation of the "Golden Test Data Set" is documented in the `create_golden_dataset.ipynb` notebook in the `PsTuts-VQA-Data-Operations` repository ([https://github.com/mbudisic/PsTuts-VQA-Data-Operations](https://github.com/mbudisic/PsTuts-VQA-Data-Operations)). This dataset (named `golden_small_hf` on Hugging Face) was then utilized in the `notebooks/evaluate_rag.ipynb` of the current project to assess the initial RAG pipeline with RAGAS.

**1. RAGAS Framework Assessment & Results 📊:**

The initial RAG pipeline ("Base" model, likely `text-embedding-3-small` before fine-tuning) yielded these mean RAGAS scores:

| Metric                          | Mean Score |
|---------------------------------|------------|
| Faithfulness                    | 0.721      |
| Answer Relevancy                | 0.914      |
| Context Recall                  | 0.672      |
| Factual Correctness (mode=f1)   | 0.654      |
| Context Entity Recall           | 0.636      |

*(Scores from `notebooks/evaluate_rag.ipynb` output for the "Base" configuration)*

**2. Conclusions on Performance and Effectiveness 🧐:**

*   **Strengths:** High **Answer Relevancy (0.914)** indicates the system understands queries well.
*   **Areas for Improvement:**
    *   **Faithfulness (0.721):** Answers are not always perfectly grounded in retrieved context.
    *   **Context Recall (0.672):** Not all necessary information is always retrieved.
    *   **Factual Correctness (0.654):** Factual accuracy of answers needs improvement.
*   **Overall:** The baseline system is good at relevant responses but needs better context retrieval and factual accuracy. This benchmarks a clear path for improvements, such as embedding fine-tuning.

# Task 6: Fine-Tuning Open-Source Embeddings

To enhance retrieval performance, an open-source embedding model was fine-tuned on domain-specific data.

**1. Fine-Tuning Process and Model Link 🔗:**

*   **Base Model:** `Snowflake/snowflake-arctic-embed-s` was chosen as the base model for fine-tuning.
*   **Fine-tuning Data:** A specialized dataset of (question, relevant_document_passage) pairs derived from the Photoshop tutorials was generated/used, as detailed in `dataset_card.md` and implemented in `notebooks/Fine_Tuning_Embedding_for_PSTuts.ipynb`.
*   **Process:** The fine-tuning was performed using the `sentence-transformers` library, with training objectives designed to improve the model's ability to map Photoshop-related queries to relevant transcript passages. The process and evaluation were tracked using W&B.
*   **Resulting Model:** The fine-tuned model was saved and pushed to the Hugging Face Hub.
*   **Hugging Face Hub Link:** The fine-tuned embedding model is available at:
    [mbudisic/snowflake-arctic-embed-s-ft-pstuts](https://huggingface.co/mbudisic/snowflake-arctic-embed-s-ft-pstuts)

*(Evidence for this is in `notebooks/Fine_Tuning_Embedding_for_PSTuts.ipynb`, specifically the `model.push_to_hub` call and its output. The `app.py` can be (or is) configured to use this fine-tuned model for the embedding step in the RAG pipeline.)*

# Task 7: Assessing Performance

Performance of the RAG application with the fine-tuned embedding model (`mbudisic/snowflake-arctic-embed-s-ft-pstuts`) was assessed using the same RAGAS framework and "Golden Data Set" (`golden_small_hf`) as the baseline.

**1. Comparative RAGAS Results 📊:** (Source: `notebooks/evaluate_rag.ipynb` output)

The notebook provides a comparison between "Base", "SOTA" (OpenAI's `text-embedding-3-small`), and "FT" (our fine-tuned `mbudisic/snowflake-arctic-embed-s-ft-pstuts`) models.

| Metric                 | Base (Initial) | FT (Fine-Tuned) | Change FT vs Base |
|------------------------|----------------|-----------------|-------------------|
| Faithfulness           | 0.721          | 0.748           | +0.027            |
| Answer Relevancy       | 0.914          | 0.819           | -0.095            |
| Context Recall         | 0.672          | 0.672           | 0.000             |
| Factual Correctness    | 0.654          | 0.598           | -0.056            |
| Context Entity Recall  | 0.636          | 0.636           | 0.000             |

*(Note: These are mean scores. `Factual Correctness` is `factual_correctness(mode=f1)` in the notebook.)*

**2. Conclusions on Fine-Tuned Performance & Future Changes 🚀:**

*   **Impact of Fine-Tuning:**
    *   **Faithfulness (+0.027):** A slight improvement, suggesting answers from the fine-tuned model are marginally more grounded in the retrieved context.
    *   **Answer Relevancy (-0.095):** Surprisingly, answer relevancy decreased. This might indicate that while the fine-tuned model is better at finding *technically* similar content based on Photoshop jargon, the overall answer framing by the LLM became less aligned with the user's original question intent compared to the broader base model.
    *   **Context Recall (No Change):** The ability to retrieve all necessary information did not change. The notebook itself notes: "What we see is that there is no difference in context recall... My guess is that this result has to do with the specific application. These were audio transcripts of fairly short videos. Most transcripts therefore fit completely into a single, or a few, chunks... even a base embedding model likely did as good of a job as it could."
    *   **Factual Correctness (-0.056):** This also saw a decrease, which is concerning and counter-intuitive for a fine-tuning step aimed at domain specificity.
*   **Overall Assessment of Fine-Tuning:** The fine-tuning of `Snowflake/snowflake-arctic-embed-s` showed mixed results. While faithfulness slightly improved, the key metrics of answer relevancy and factual correctness unexpectedly declined. Context recall remained unchanged, which the notebook speculates might be due to the nature of the data (short, distinct transcripts). The notebook author concludes: "So, in the end, the conclusion is that the embedding model is not the right spot to optimize this RAG chain." for this specific dataset and base embedding model.
*   **Expected Changes & Future Improvements:**
    1.  **Re-evaluate Fine-Tuning Strategy:** Given the results, the fine-tuning approach for embeddings needs review. This could involve:
        *   Trying a different base model for fine-tuning (perhaps a larger one, or one known for better transfer learning on smaller datasets).
        *   Augmenting the fine-tuning dataset or using different data generation strategies.
        *   Adjusting fine-tuning hyperparameters.
    2.  **Prompt Engineering:** Focus on refining the prompts used for the LLM agents (supervisor, RAG agent) to better guide answer synthesis, potentially improving factual correctness and answer relevancy irrespective of embedding model changes.
    3.  **Advanced RAG Techniques:** Explore techniques like re-ranking retrieved documents, query transformations, or hypothetical document embeddings (HyDE) to improve the quality and relevance of context fed to the LLM.
    4.  **LLM for Generation:** Experiment with different LLMs for the answer generation step. The `evaluate_rag.ipynb` uses `gpt-4.1-nano` for the LLM in RAG chains and `gpt-4.1-mini` for the evaluator LLM. The main `app.py` uses `gpt-4.1-mini`. Consistency or using a more powerful generation model might yield better results.
    5.  **Iterative Evaluation:** Continue using the RAGAS framework on the golden dataset to meticulously track the impact of each change.

This concludes the update to `ANSWER.md` based on your instructions. 