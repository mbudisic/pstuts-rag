## 🚀 Enhancing RAG: A Look at Our Initial Chunking Strategy and Timestamp Alignment 🧠

Hello everyone. 👋 If you've worked with Retrieval Augmented Generation (RAG), you might have encountered situations where the AI's responses, while promising, didn't quite hit the mark. Perhaps they lacked specificity, missed crucial nuances, or included irrelevant information. A common factor influencing this is the **chunking strategy** – how we break down source documents for the model.

RAG significantly expands the capabilities of Large Language Models (LLMs) by allowing them to access custom knowledge bases. However, the effectiveness of the "retrieval" phase is heavily dependent on how this knowledge is segmented, or "chunked." An unsuitable chunking approach can lead to confusing or incomplete information being fed to the LLM. Conversely, a well-thought-out strategy can greatly improve precision and relevance.

This post will walk through the initial chunking methodology implemented in our [PsTuts RAG project](https://github.com/mbudisic/pstuts-rag/blob/main/). This system is designed to answer Adobe Photoshop queries using video tutorial transcripts. We'll examine why basic chunking can be insufficient and how our first iteration of semantic chunking, with a strong emphasis on aligning with original video timestamps, aims to provide a more robust solution. 💡

---

### 🤔 The Challenge of Chunking: Balancing Size and Meaning

Consider how you would process a detailed instructional video. Would you prefer:

1.  Arbitrarily segmented, fixed-length transcript portions, potentially interrupting sentences or ideas?
2.  Complete thoughts or distinct instructional steps, even if their textual length varies?

Most would likely opt for the second choice. LLMs, in their own way, benefit from similarly coherent inputs.

**Common Chunking Methods & Their Limitations:**

*   **Fixed-Size Chunking:** This approach divides text into segments of a predefined length (e.g., 200 words). While straightforward to implement, it often results in fragmented ideas, as natural semantic boundaries are ignored. Context can be lost, and the resulting chunks may not represent complete thoughts.
*   **Sentence Splitting:** Dividing text by sentences is an improvement. However, a single sentence may not always encapsulate a full idea, particularly in complex material. Furthermore, several sentences might be closely related, forming a single, cohesive semantic unit that is best kept together.
*   **Paragraph Splitting:** This method gets closer to ideal, as paragraphs often group related ideas. However, a single paragraph might still cover multiple distinct sub-topics, or one complex idea might be articulated across several shorter paragraphs.

The fundamental issue with these methods is their reliance on structural cues rather than semantic content. They don't deeply analyze the *meaning* of the text being divided. For a RAG system, this can mean that retrieved chunks offer only partial views of the necessary information or mix relevant details with irrelevant ones, leading to less than optimal LLM outputs.

---

### ✨ Introducing Semantic Chunking: Prioritizing Coherent Context

Semantic chunking aims to divide text based on its underlying meaning. Instead of relying on fixed lengths or simple punctuation, this method seeks to identify natural breakpoints where topics shift or distinct concepts are concluded. The objective is to produce chunks that are internally consistent and rich in context.

This is like identifying distinct "scenes" in a film script, rather than just cutting it every X lines. Each scene (or semantic chunk) carries a more complete segment of the narrative.

For the PsTuts RAG project, which uses video tutorial transcripts, this approach is particularly important. A Photoshop tutorial often involves multi-step processes. A semantic chunk ideally encapsulates one complete step, making it a well-defined unit of information for the LLM.

**Conceptual Basis:**
Semantic chunking frequently employs embedding models. These models transform text into numerical vectors that represent its meaning. By assessing the similarity between vectors of adjacent sentences or sentence groups, the system can detect points where semantic similarity decreases, suggesting a topic shift and a suitable location for a chunk boundary.

The [Langchain library](https://python.langchain.com/docs/get_started/introduction), a comprehensive framework for developing LLM applications, provides tools such as the `SemanticChunker`, which we utilize in our system.

---

### 🛠️ Our First Iteration: Semantic Chunking with Timestamp Preservation

Let's examine the specifics of our initial implementation for the PsTuts video transcripts. A primary driver for this first version of our chunking strategy was the need to connect semantically coherent text segments back to their precise timings in the original videos. This feature is highly beneficial for users who may wish to navigate directly to the relevant moment in a tutorial.

You can explore the implementation details in our GitHub repository: [`mbudisic/pstuts-rag`](https://github.com/mbudisic/pstuts-rag/blob/main/).

**Step 1: Loading the Source Data 📜**

The process begins with ingesting our data. The video transcripts are initially in JSON format, with each entry containing a spoken sentence and its corresponding start and end timestamps.

We employ two primary loaders, located in `pstuts_rag/pstuts_rag/loader.py` ([view on GitHub](https://github.com/mbudisic/pstuts-rag/blob/main/pstuts_rag/pstuts_rag/loader.py)):

1.  `VideoTranscriptChunkLoader`: This loader processes the JSON input to create individual `Document` objects for each sentence (or small verbatim segment from the transcript). Critically, it retains the `time_start` and `time_end` metadata for every sentence. These serve as our fundamental temporal reference points.

    ```python
    # Excerpt from VideoTranscriptChunkLoader in loader.py
    # ...
    # for transcript in transcripts:
    #     yield Document(
    #         page_content=transcript["sent"],
    #         metadata=metadata
    #         | {
    #             "time_start": transcript["begin"],
    #             "time_end": transcript["end"],
    #         },
    #     )
    ```

2.  `VideoTranscriptBulkLoader`: This loader adopts a broader perspective. For each video, it concatenates all its sentences into a single `Document`. This provides the complete, continuous text of each tutorial, which serves as the input for the semantic chunker. Feeding the chunker individual sentences would deprive it of the wider context needed to identify meaningful breakpoints that span multiple sentences.

    ```python
    # Excerpt from VideoTranscriptBulkLoader in loader.py
    # ...
    # yield Document(
    #     page_content="\n".join(
    #         t["sent"] for t in video["transcripts"]
    #     ),
    #     metadata=metadata,
    # )
    ```

This dual loading approach provides two representations of our data: a fine-grained, sentence-level view with precise timestamps, and a comprehensive, full-transcript view suitable for semantic division.

**Step 2: Semantic Segmentation 🧠🔪**

This stage is handled primarily by the `chunk_transcripts` function in `pstuts_rag/pstuts_rag/datastore.py` ([view on GitHub](https://github.com/mbudisic/pstuts-rag/blob/main/pstuts_rag/pstuts_rag/datastore.py)).

We provide the `docs_full_transcript` (from `VideoTranscriptBulkLoader`) to Langchain's `SemanticChunker`. We utilize `OpenAIEmbeddings` (specifically `text-embedding-3-small`) for this process, as these embeddings enable the chunker to interpret the semantic content of the text.

```python
# Excerpt from chunk_transcripts in datastore.py
# ...
# text_splitter = SemanticChunker(semantic_chunker_embedding_model)
# docs_group = await asyncio.gather(
#     *[
#         text_splitter.atransform_documents(d)
#         for d in batch(docs_full_transcript, size=2) # Batching for efficiency
#     ]
# )
# # Flatten the nested list of documents
# docs_chunks_semantic: List[Document] = []
# for group in docs_group:
#     docs_chunks_semantic.extend(group)
# ...
```

The `SemanticChunker` intelligently divides the long transcript of each video into a series of smaller, semantically related chunks. Each of these `docs_chunks_semantic` aims to represent a distinct idea or step from the tutorial. The user described this goal as creating "semantic Kamradt chunks"— a term reflecting the aspiration for these idea-units, though "Kamradt chunking" itself isn't a standard Langchain term.

At this stage, our semantic chunks are textually defined. However, they lack a crucial piece of information for video-based content: **timing**. A user inquiring, "How do I use the clone stamp tool?" would benefit not only from the textual explanation but also from knowing *where* in the video that explanation is located.

**Step 3: Aligning Semantic Chunks with Timestamps 🕰️🔗**

This is a key part of our initial strategy: establishing a connection between the semantically defined chunks and the original, timestamped sentences. The goal is to determine which of our original, granular sentences (from `VideoTranscriptChunkLoader`) constitute each new semantic chunk.

Within the `chunk_transcripts` function (`datastore.py`), we iterate through each semantic chunk (`docs_chunks_semantic`). For each one, we reference our collection of original, timestamped sentences (`docs_chunks_verbatim`):

```python
# Excerpt from chunk_transcripts in datastore.py
# ...
# # Create a lookup dictionary for faster access to verbatim chunks by video_id
# video_id_to_chunks: Dict[int, List[Document]] = {}
# for chunk_v in docs_chunks_verbatim:
#     video_id: int = chunk_v.metadata["video_id"]
#     if video_id not in video_id_to_chunks:
#         video_id_to_chunks[video_id] = []
#     video_id_to_chunks[video_id].append(chunk_v)

# for chunk_s in docs_chunks_semantic: # Our new semantic chunk
#     video_id = chunk_s.metadata["video_id"]
#     # Only check verbatim chunks from the same video
#     potential_subchunks = video_id_to_chunks.get(video_id, [])
#     subchunks = [
#         c
#         for c in potential_subchunks
#         if c.page_content in chunk_s.page_content # Direct search for sentence text
#     ]
# ...
```

The line `if c.page_content in chunk_s.page_content` is pivotal. It operates on the premise that the text of an original, timestamped sentence will be present within the text of the larger semantic chunk it belongs to. This "direct search" is effective because the semantic chunk is typically a concatenation or superset of several original sentences.

After identifying all original sentences (`subchunks`) that comprise a given semantic chunk, we extract their timestamps:

```python
# Excerpt from chunk_transcripts in datastore.py
# ...
#     times = [
#         (t.metadata["time_start"], t.metadata["time_end"])
#         for t in subchunks
#     ]
#     chunk_s.metadata["speech_start_stop_times"] = times # Store all individual sentence times

#     if times:  # Check if times list is non-empty
#         chunk_s.metadata["start"], chunk_s.metadata["stop"] = (
#             times[0][0],    # Start time of the first sentence in the semantic chunk
#             times[-1][-1],  # End time of the last sentence in the semantic chunk
#         )
#     else:
#         chunk_s.metadata["start"], chunk_s.metadata["stop"] = None, None
# ...
```

As a result, each semantic chunk is enriched with:
*   `speech_start_stop_times`: A list of (start, end) time tuples for every original sentence it incorporates.
*   `start`: The start time of the very first sentence within the semantic chunk.
*   `stop`: The end time of the very last sentence within the semantic chunk.

This metadata is highly valuable. When our RAG system retrieves a semantic chunk, it receives not only a coherent piece of information but also precise timing data, allowing the user to navigate directly to that segment in the source video.

**Step 4: Preparing for Retrieval by Storing in a Vector Database 💾**

Once our documents are chunked and timestamped, they are vectorized (again, using `OpenAIEmbeddings`) and stored in our chosen vector database, Qdrant. This process is overseen by the `DatastoreManager` class, also found in `pstuts_rag/pstuts_rag/datastore.py`.

```python
# Excerpt from DatastoreManager in datastore.py
# ...
# async def populate_database(self, raw_docs: List[Dict[str, Any]]) -> int:
#     # Perform chunking (which includes timestamp association)
#     self.docs: List[Document] = await chunk_transcripts(
#         json_transcripts=raw_docs,
#         semantic_chunker_embedding_model=self.embeddings,
#     )
#     # ... then perform embedding and upload to Qdrant ...
# ...
```

The `DatastoreManager` is responsible for generating vector embeddings for these enriched chunks and indexing them in Qdrant, making them efficiently searchable for the RAG system.

---

### 🏆 Benefits of This Initial Approach

Why adopt this specific methodology for our first iteration?

1.  **Improved Relevance:** Semantic chunks aim to provide LLMs with more complete and contextually sound information. This can lead to more relevant and accurate responses compared to simpler chunking methods.
2.  **Enhanced User Navigation:** For video content, associating chunks with timestamps is a significant usability improvement. Users can be directed to the precise moment in a tutorial where the information is presented, saving time and effort.
3.  **Efficient Use of Context Window:** LLMs operate with a finite context window. Semantically coherent chunks help make better use of this limited space by providing meaningful information rather than fragmented text.
4.  **Foundation for Reduced Errors:** When LLMs receive better, more focused context, they may be less prone to generating incorrect or unsupported information. The retrieved chunks serve as stronger grounding.
5.  **Adaptability for Complex Material:** As the length and complexity of source documents (or videos) grow, the advantages of a semantic approach to chunking generally become more apparent.

In the PsTuts RAG system, this initial strategy means that user queries can be answered with text that not only reflects the tutorial content accurately but is also linked directly to the corresponding segments in the video.

---

### 🤔 Points to Consider and Future Refinements

While this initial approach offers advantages, there are several aspects to consider for ongoing development:

*   **Computational Resources:** Semantic chunking, particularly when using embeddings, generally requires more computational resources upfront compared to fixed-size splitting. The use of `asyncio` and batch processing in our implementation helps manage this.
*   **Choice of Embedding Model:** The effectiveness of semantic chunking is influenced by the chosen embedding model. Our current use of `text-embedding-3-small` represents a balance between performance and cost; however, different datasets or requirements might benefit from exploring other models.
*   **Chunker Parameter Tuning:** The `SemanticChunker` in Langchain offers parameters (such as breakpoint thresholds) that can be adjusted. Optimizing these for specific datasets may involve some experimentation.
*   **Assumption of Textual Containment:** The timestamp association method relies on the `page_content` of the original sentences being directly present within the `page_content` of the semantic chunk. This generally holds for how `SemanticChunker` aggregates text. However, alternative chunking methods that perform more aggressive summarization might require a different association logic.
*   **Iteration and Evaluation:** This is our first iteration of the chunking strategy. Future work will involve evaluating its performance with robust metrics (e.g., RAGAS) and exploring alternative or more advanced chunking techniques to further enhance retrieval quality.

---

### 🎬 Conclusion: A Pragmatic First Step in Chunking for RAG

The method used to prepare and structure data for RAG systems is a critical determinant of their performance. While simpler chunking techniques are quicker to implement, investing in a more meaning-oriented strategy like semantic chunking, especially when combined with essential domain-specific features like timestamp association, can significantly improve the utility of a RAG application.

For developers building RAG systems, particularly those dealing with multimedia or structured content where source linking and contextual integrity are important, it is worthwhile to look beyond basic splitting methods. Exploring semantic chunking and carefully considering how to preserve and leverage metadata can lead to a more effective and user-friendly system.

Our initial approach for the PsTuts RAG project, focusing on semantic coherence and robust timestamp linking, represents a pragmatic first step. The journey from raw transcript to a usefully chunked and timestamped piece of information in our vector store involves several coordinated steps, each aimed at maximizing the relevance and utility of the data for the end-user.

We encourage you to review the implementation in the [PsTuts RAG project on GitHub](https://github.com/mbudisic/pstuts-rag/blob/main/). As we continue to develop this project, we will be exploring further refinements to this process.

What are your experiences with chunking for RAG? Any particular challenges or successes you'd like to share? We welcome your thoughts in the comments section. 👇

#RAG #AI #LLM #SemanticChunking #VectorDatabase #Qdrant #Langchain #Python #Developer #DataScience #MachineLearning #PsTutsRAG

---
*Note: The term "Kamradt chunking" was used by the project owner to describe the desired outcome of creating semantically coherent, idea-based chunks, similar to the objective of Langchain's `SemanticChunker`.*
