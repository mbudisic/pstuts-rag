# Certification Challenge

Marko Budisic

## Deliverables

1. [Main Github repo](https://github.com/mbudisic/pstuts-rag)
2. [Github repo for creating the golden dataset](https://github.com/adobe-research/PsTuts-VQA-Dataset)
3. [Loom video]()
4. [Written document](https://github.com/mbudisic/pstuts-rag/blob/main/ANSWER.md)
5. [Hugging Face live demo](https://huggingface.co/spaces/mbudisic/PsTuts-RAG)
6. [Fine tuned embedding model](https://huggingface.co/mbudisic/snowflake-arctic-embed-s-ft-pstuts)
7. [Corpus dataset](https://huggingface.co/datasets/mbudisic/PsTuts-VQA)
8. [Golden Q&A dataset](https://huggingface.co/datasets/mbudisic/pstuts_rag_qa)

## ToC

- [Certification Challenge](#certification-challenge)
  - [Deliverables](#deliverables)
  - [ToC](#toc)
- [Task 1: Defining your Problem and Audience](#task-1-defining-your-problem-and-audience)
- [Task 2: Propose a Solution](#task-2-propose-a-solution)
- [Task 3: Dealing with the Data](#task-3-dealing-with-the-data)
  - [3.1. Data Sources \& External APIs 📊+🌐:](#31-data-sources--external-apis-)
  - [3.2. Chunking Strategy 🧠✂️:](#32-chunking-strategy-️)
  - [3.3. Specific Data Needs for Other Parts 🧩:](#33-specific-data-needs-for-other-parts-)
- [Task 4: Building a Quick End-to-End Prototype](#task-4-building-a-quick-end-to-end-prototype)
  - [4.1. The Prototype Application 🖥️:](#41-the-prototype-application-️)
  - [4.2. Deployment 🚀 (Hugging Face Space):](#42-deployment--hugging-face-space)
- [Task 5: Creating a Golden Test Data Set](#task-5-creating-a-golden-test-data-set)
  - [5.1. RAGAS Framework Assessment \& Results 📊:](#51-ragas-framework-assessment--results-)
- [Task 6: Fine-Tuning Open-Source Embeddings](#task-6-fine-tuning-open-source-embeddings)
  - [6.1. Fine-Tuning Process and Model Link 🔗:\*\*](#61-fine-tuning-process-and-model-link-)
- [Task 7: Assessing Performance](#task-7-assessing-performance)
  - [7.1. Comparative RAGAS Results 📊:](#71-comparative-ragas-results-)
  - [7.2. Conclusions on Fine-Tuned Performance  🚀:](#72-conclusions-on-fine-tuned-performance--)
- [8. Future changes](#8-future-changes)


# Task 1: Defining your Problem and Audience

**Problem:** Navigating extensive libraries of video materials to find specific information is often a time-consuming and inefficient process for users. This challenge is common in organizations that rely on video-based training materials. 😓

**Users and their Problem:** 🏢 Companies often have extensive video tutorial libraries for proprietary software. Employees (new hires, support, experienced users) struggle to quickly find specific instructions within these videos. 🎯 Like Photoshop learners needing a specific technique, employees need a fast way to query video content, saving time and boosting learning. 🚀

_Side note: This is a good approximation of a problem that I am internally solving for my company. The agentic RAG will be augmented further for the demo day._

# Task 2: Propose a Solution

**Our Solution:** 🗣️ An agentic Retrieval Augmented Generation (RAG) system designed to answer questions about a company's video tutorial library (e.g., for software like Adobe Photoshop, or any internal training content). Users interact via a chat interface (Chainlit, as seen in `app.py`). 💻 The system queries its knowledge base of tutorial transcripts and can use Tavily for web searches, providing comprehensive answers relevant to the specific video library and serving up videos at the referenced timestampes. 🌐

Broader vision is to build an ingestion pipeline that would transcribe audio narration and OCR
key frames in the video to further enhance the context.
Users would be able to search not only by a query, but also by a screenshot (e.g. looking up
live video if they have only a screenshot in a company walkthrough).
The agents would not only be able to answer the queries, but also develop a 
short presentation, e.g., in `reveal.js` or `remark`.

**The Tech Stack 🛠️:**

*   **LLM:** OpenAI model (`gpt-4.1-mini`), selected for strong language capabilities and ease of API access. 🧠
*   **Embedding Model:** An open-source model, `Snowflake/snowflake-arctic-embed-s` (see `Fine_Tuning_Embedding_for_PSTuts.ipynb`), fine-tuned for domain-specific relevance. This is a small model trainable on a laptop. ❄️
*   **Orchestration:** LangChain & LangGraph, for building the RAG application and managing agent workflows. Many functions have been stored in the `pstuts_rag` package to allow calling from notebooks and app. 🔗
*   **Vector Database:** Qdrant (`pstuts_rag/datastore.py`), for efficient semantic search of tutorial transcripts. I had most experience with it, and no reason to look elsewhere. 💾
*   **Evaluation:** Synthetic data set, [created using RAGAS in a second repo](https://github.com/mbudisic/PsTuts-VQA-Data-Operations), powers `evaluate_rag.ipynb`, for assessing RAG pipeline (w/o the search powers) quality. 🧐
*   **Monitoring:** W&B (Weights & Biases)🏋️ was used to monitor fine-tuning. LangSmith was enabled for monitoring in general.📊
*   **User Interface:** Chainlit chat with on-demand display of videos positioned at the correct timestamp. 💬 📼
*   **Serving & Inference:** Docker (`Dockerfile`), for containerized deployment on Hugging Face Spaces. 🐳

**The Role of Agents 🕵️‍♂️:** 

The system uses a LangGraph-orchestrated multi-agent approach:
1.  **Supervisor Agent:** Manages the overall workflow. It receives the user query and routes it to the appropriate specialized agent based on its interpretation of the query (defined in `SUPERVISOR_SYSTEM` prompt and `create_team_supervisor` in `app.py`). 🧑‍✈️
2.  **Video Archive Agent (`VIDEOARCHIVE`):** This is the RAG agent. It queries the Qdrant vector store of Photoshop tutorial transcripts to find relevant information and generates an answer based on this retrieved context. (Uses `create_rag_node` from `pstuts_rag.agent_rag`). 📼
3.  **Adobe Help Agent (`ADOBEHELP`):** This agent uses the Tavily API to perform web searches if the supervisor deems it necessary for broader or more current information. (Uses `create_tavily_node` from `pstuts_rag.agent_tavily`). 🌍
The supervisor then determines if the answer is complete or if further steps are needed. ✅

```
                            +-----------+                             
                            | __start__ |                             
                            +-----------+                             
                                  *                                   
                                  *                                   
                                  *                                   
                            +------------+                            
                            | supervisor |                            
                       *****+------------+.....                       
                   ****            .           ....                   
              *****                .               .....              
           ***                     .                    ...           
+-----------+           +--------------------+           +---------+  
| AdobeHelp |           | VideoArchiveSearch |           | __end__ |  
+-----------+           +--------------------+           +---------+  
```

# Task 3: Dealing with the Data

## 3.1. Data Sources & External APIs 📊+🌐:

*   **Primary Data:** [PsTuts-VQA](https://github.com/adobe-research/PsTuts-VQA-Dataset) is a publicly-released set of transcripts linked to a database of Adobe-created Photoshop training videos. Data is in a JSON format, made available on [hf.co:mbudisic/PsTuts-VQA](https://huggingface.co/datasets/mbudisic/PsTuts-VQA). 📁
*   **External API:** Tavily Search API (configured in `app.py`) augments knowledge with web search results of domain [helpx.adobe.com](https://helpx.adobe.com) via the `ADOBEHELP` agent for current or broader topics not covered in the internal videos. 🔍

## 3.2. Chunking Strategy 🧠✂️:

(see: `pstuts_rag/datastore.py`'s `chunk_transcripts` function and `pstuts_rag/loader.py`)

Transcript chunks in the input dataset are too granular - often a sentence or two,
since they are tied to the time windows in which a particular transcript sentence would
be overlaid on the screen.

Therefore, to achieve a useful semantic chunking for RAG, the following **semantic chunking** strategy is employed:
1.  **Initial Loading:** Transcripts are loaded both entirely per video (`VideoTranscriptBulkLoader`) and as individual sentences/segments with timestamps (`VideoTranscriptChunkLoader`).
2.  **Semantic Splitting:** `SemanticChunker` (LangChain, using `OpenAIEmbeddings`) splits full transcripts into semantically coherent chunks.
3.  **Metadata Enrichment:** These semantic chunks are enriched with start/end times by mapping them back to the original timestamped sentences.

  **In summary:** 🤔 This method (a) creates topically focused chunks for better retrieval. 🎯 (b) links back to video timestamps. 🔗

## 3.3. Specific Data Needs for Other Parts 🧩:

*   **Evaluation & Golden Dataset (Tasks 5 & 7):** 🏆 Generating the "Golden Data Set" (Q-C-A triplets) for RAGAS is detailed in `create_golden_dataset.ipynb` (see [`PsTuts-VQA-Data-Operations` repo](https://github.com/mbudisic/PsTuts-VQA-Data-Operations)). The resulting dataset [hf.co:mbudisic/pstuts_rag_-_qa](https://huggingface.co/datasets/mbudisic/pstuts_rag_qa) is used to benchmark the RAG pipeline in `evaluate_rag.ipynb`. 📊

*   **Embedding Model Fine-Tuning (Task 6):** 🔬 The `Fine_Tuning_Embedding_for_PSTuts.ipynb` notebook shows the use of [hf.co:mbudisic/pstuts_rag_-_qa](https://huggingface.co/datasets/mbudisic/pstuts_rag_qa) tp fine-tune the embedding model. This adapts models like `Snowflake/snowflake-arctic-embed-s` for improved retrieval. ⚙️

# Task 4: Building a Quick End-to-End Prototype

An end-to-end prototype RAG system for Photoshop tutorials is built and deployed to HF.

## 4.1. The Prototype Application 🖥️:

The `app.py` script is the core prototype. It uses Chainlit for the UI, LangChain/LangGraph for orchestration, Qdrant for the vector store, and OpenAI models for embeddings and generation. It loads data, builds the RAG chain, and manages the agentic workflow for user queries. ✨

## 4.2. Deployment 🚀 (Hugging Face Space):

The repository is structured for Hugging Face Space deployment:
*   `README.md` contains Hugging Face Space metadata (e.g., `sdk: docker`).
*   A `Dockerfile` enables containerization for deployment.
This setup indicates the prototype is packaged for public deployment. 🌍

# Task 5: Creating a Golden Test Data Set

The creation of the "Golden Test Data Set" is documented in the `create_golden_dataset.ipynb` notebook in the `PsTuts-VQA-Data-Operations` repository ([https://github.com/mbudisic/PsTuts-VQA-Data-Operations](https://github.com/mbudisic/PsTuts-VQA-Data-Operations)). This dataset (named `golden_small_hf` on Hugging Face) was then utilized in the `notebooks/evaluate_rag.ipynb` of the current project to assess the initial RAG pipeline with RAGAS. 🌟

## 5.1. RAGAS Framework Assessment & Results 📊:

The initial RAG pipeline ("Base" model, `Snowflake/snowflake-arctic-embed-s` before fine-tuning) yielded these mean RAGAS scores:

| Metric                          | Mean Score |
|---------------------------------|------------|
| Faithfulness                    | 0.721      |
| Answer Relevancy                | 0.914      |
| Context Recall                  | 0.672      |
| Factual Correctness (mode=f1)   | 0.654      |
| Context Entity Recall           | 0.636      |

*(Scores from `notebooks/evaluate_rag.ipynb` output for the "Base" configuration)*

**2. Conclusions on Performance and Effectiveness 🧐:**

*   **Strengths:** 💪 High **Answer Relevancy (0.914)** indicates the system understands queries well.
*   **Areas for Improvement:** 📉
    *   **Faithfulness (0.721):** Answers are not always perfectly grounded in retrieved context.
    *   **Context Recall (0.672):** Not all necessary information is always retrieved.
    *   **Factual Correctness (0.654):** Factual accuracy of answers needs improvement.
*   **Overall:** The baseline system is good at relevant responses but needs better context retrieval and factual accuracy. This benchmarks a clear path for improvements, such as embedding fine-tuning. 🛠️

# Task 6: Fine-Tuning Open-Source Embeddings

To enhance retrieval performance for a specific video library, an open-source embedding model can be fine-tuned on domain-specific data. The following describes an example of this process using Photoshop tutorial data. 🧪

## 6.1. Fine-Tuning Process and Model Link 🔗:**

*   **Base Model:** `Snowflake/snowflake-arctic-embed-s` was chosen as the base model for fine-tuning in this example. ❄️
*   **Fine-tuning Data:** For this specific example, a specialized dataset of (question, relevant_document_passage) pairs derived from Photoshop tutorials was generated/used, as detailed in `dataset_card.md` and implemented in `notebooks/Fine_Tuning_Embedding_for_PSTuts.ipynb`. A similar dataset would be created for any other specific domain. 🖼️
*   **Process:** 🛠️ Fine-tuning used `sentence-transformers` to better map domain queries (e.g., Photoshop) to transcript passages. W&B tracked the process and evaluation. 📈
*   **Resulting Model:** The fine-tuned model (for the Photoshop example) was saved and pushed to the Hugging Face Hub. 🤗
*   **Hugging Face Hub Link (Example):** The fine-tuned embedding model for the Photoshop tutorial example is available at:
    [mbudisic/snowflake-arctic-embed-s-ft-pstuts](https://huggingface.co/mbudisic/snowflake-arctic-embed-s-ft-pstuts)

*(Evidence for this is in `notebooks/Fine_Tuning_Embedding_for_PSTuts.ipynb`, specifically the `model.push_to_hub` call and its output. The `app.py` can be (or is) configured to use such a fine-tuned model for the embedding step in the RAG pipeline.)*

# Task 7: Assessing Performance

Performance of the RAG application with the fine-tuned embedding model (`mbudisic/snowflake-arctic-embed-s-ft-pstuts`) was assessed using the same RAGAS framework and "Golden Data Set" (`golden_small_hf`) as the baseline. 🏆

## 7.1. Comparative RAGAS Results 📊:

(see: `notebooks/evaluate_rag.ipynb` output)

The notebook provides a comparison between "Base", "SOTA" (OpenAI's `text-embedding-3-small`), and "FT" (our fine-tuned `mbudisic/snowflake-arctic-embed-s-ft-pstuts`) models.

| Metric                 | Base (Initial) | FT (Fine-Tuned) | Change FT vs Base |
|------------------------|----------------|-----------------|-------------------|
| Faithfulness           | 0.721          | 0.748           | +0.027            |
| Answer Relevancy       | 0.914          | 0.819           | -0.095            |
| Context Recall         | 0.672          | 0.672           | 0.000             |
| Factual Correctness    | 0.654          | 0.598           | -0.056            |
| Context Entity Recall  | 0.636          | 0.636           | 0.000             |

*(Note: These are mean scores. `Factual Correctness` is `factual_correctness(mode=f1)` in the notebook.)*

## 7.2. Conclusions on Fine-Tuned Performance  🚀:

*   **Impact of Fine-Tuning:**
    *   **Faithfulness (+0.027):** ✅ A slight improvement, suggesting answers from the fine-tuned model are marginally more grounded in the retrieved context.
    *   **Answer Relevancy (-0.095):** 📉 Surprisingly, relevancy decreased. While the FT model found technically similar content (e.g., Photoshop jargon), the LLM's answer framing may have become less aligned with user intent versus the base model.
    *   **Context Recall (No Change):** 🤷‍♂️ Retrieval ability remained static. The notebook suggests this might be due to short video transcripts fitting into few chunks, where even base embeddings perform well.
    *   **Factual Correctness (-0.056):** 📉 This also saw a decrease, which is concerning and counter-intuitive for a fine-tuning step aimed at domain specificity.
*   **Overall Assessment of Fine-Tuning:** 🤔 Mixed results for `Snowflake/snowflake-arctic-embed-s` fine-tuning. Faithfulness slightly up, but answer relevancy and factual correctness surprisingly dropped. Context recall was unchanged (likely due to data nature). The notebook concludes embedding model tuning isn't the prime optimization spot here. 🎯
*   

# 8. Future changes

*   **Expected Changes & Future Improvements:**
    1.  **Re-evaluate Fine-Tuning Strategy: 🤔** Given results, embedding fine-tuning needs review. This could involve:
        *   Trying a different base model (larger, better transfer learning on small datasets).
        *   Augmenting the fine-tuning dataset or using different data generation strategies.
        *   Adjusting fine-tuning hyperparameters.
    2.  **Prompt Engineering: ✍️** Refine LLM agent prompts (supervisor, RAG) for better answer synthesis. This could boost factual correctness and relevancy, regardless of embedding model.
    3.  **Advanced RAG Techniques: ✨** Explore methods like re-ranking, query transformations, or HyDE. The goal is to improve context quality and relevance for the LLM.
    4.  **LLM for Generation: 🧠** Experiment with different LLMs for answer generation. `evaluate_rag.ipynb` uses `gpt-4.1-nano` (RAG) and `gpt-4.1-mini` (evaluator); `app.py` uses `gpt-4.1-mini`. Consistency or a more powerful model might improve results.
    5.  **Iterative Evaluation: 🔁** Keep using RAGAS on the golden dataset. This will meticulously track each change's impact.

This concludes the update to `ANSWER.md` based on your instructions. 