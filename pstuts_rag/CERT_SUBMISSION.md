# Task 1: Defining your Problem and Audience

**You are an AI Solutions Engineer**.

**What** problem do you want to solve?  **Who** is it a problem for?

<aside>
📝

Task 1: Articulate the problem and the user of your application

*Hints:* 

- *Create a list of potential questions that your user is likely to ask!*
- *What is the user’s job title, and what is the part of their job function that you’re trying to automate?*
</aside>

**✅ Deliverables**

1. Write a succinct 1-sentence description of the problem
2. Write 1-2 paragraphs on why this is a problem for your specific user

<aside>
⚠️

**If you cannot come up with a problem worth solving, use this one as a default**.

**⚖️ Default Problem**: *Legal documents are too hard to understand for average people*

Default Solution: Build a fine-tuned, agentic RAG application that can answer questions in simple language about a court case based on source documents and additional relevant retrieved information

</aside>

# Task 2: Propose a Solution

Now that you’ve defined a problem and a user, *there are many possible solutions*.

Choose one, and articulate it.

<aside>
📝

Task 2: Articulate your proposed solution

*Hint:*  

- *Paint a picture of the “better world” that your user will live in.  How will they save time, make money, or produce higher-quality output?*
- *Recall the [LLM Application stack](https://a16z.com/emerging-architectures-for-llm-applications/) we’ve discussed at length*
</aside>

**✅ Deliverables**

1. Write 1-2 paragraphs on your proposed solution.  How will it look and feel to the user?
2. Describe the tools you plan to use in each part of your stack.  Write one sentence on why you made each tooling choice.
    1. LLM
    2. Embedding Model
    3. Orchestration
    4. Vector Database
    5. Monitoring
    6. Evaluation
    7. User Interface
    8. (Optional) Serving & Inference
3. Where will you use an agent or agents?  What will you use “agentic reasoning” for in your app?

# Task 3: Dealing with the Data

**You are an AI Systems Engineer.**  The AI Solutions Engineer has handed off the plan to you.  Now *you must identify some source data* that you can use for your application.  

Assume that you’ll be doing at least RAG (e.g., a PDF) with a general agentic search (e.g., a search API like [Tavily](https://tavily.com/) or [SERP](https://serpapi.com/)).

Do you also plan to do fine-tuning or alignment?  Should you collect data, use Synthetic Data Generation, or use an off-the-shelf dataset from [HF Datasets](https://huggingface.co/docs/datasets/en/index) or [Kaggle](https://www.kaggle.com/datasets)?

<aside>
📝

Task 3: Collect data for (at least) RAG and choose (at least) one external API

*Hint:*  

- *Ask other real people (ideally the people you’re building for!) what they think.*
- *What are the specific questions that your user is likely to ask of your application?  **Write these down**.*
</aside>

**✅ Deliverables**

1. Describe all of your data sources and external APIs, and describe what you’ll use them for.
2. Describe the default chunking strategy that you will use.  Why did you make this decision?
3. [Optional] Will you need specific data for any other part of your application?   If so, explain.

# Task 4: Building a Quick End-to-End Prototype

<aside>
📝

Task 4: Build an end-to-end RAG application using an industry-standard open-source stack and your choice of commercial off-the-shelf models

</aside>

**✅ Deliverables**

1. Build an end-to-end prototype and deploy it to a Hugging Face Space (or other endpoint)

# Task 5: Creating a Golden Test Data Set

**You are an AI Evaluation & Performance Engineer.**  The AI Systems Engineer who built the initial RAG system has asked for your help and expertise in creating a "Golden Data Set" for evaluation.

<aside>
📝

Task 5: Generate a synthetic test data set to baseline an initial evaluation with RAGAS

</aside>

**✅ Deliverables**

1. Assess your pipeline using the RAGAS framework including key metrics faithfulness, response relevance, context precision, and context recall.  Provide a table of your output results.
2. What conclusions can you draw about the performance and effectiveness of your pipeline with this information?

# Task 6: Fine-Tuning Open-Source Embeddings

**You are a Machine Learning Engineer.**  The AI Evaluation and Performance Engineer has asked for your help to fine-tune the embedding model.

<aside>
📝

Task 6: Generate synthetic fine-tuning data and complete fine-tuning of the open-source embedding model

</aside>

**✅ Deliverables**

1. Swap out your existing embedding model for the new fine-tuned version.  Provide a link to your fine-tuned embedding model on the Hugging Face Hub.

# Task 7: Assessing Performance

**You are the AI Evaluation & Performance Engineer**.  It's time to assess all options for this product.

<aside>
📝

Task 7: Assess the performance of the fine-tuned agentic RAG application

</aside>

**✅ Deliverables**

1. How does the performance compare to your original RAG application?  Test the fine-tuned embedding model using the RAGAS frameworks to quantify any improvements.  Provide results in a table.
2. Articulate the changes that you expect to make to your app in the second half of the course. How will you improve your application?

# Your Final Submission

Please include the following in your final submission:

1. A public (or otherwise shared) link to a **GitHub repo** that contains:
    1. A 5-minute (OR LESS) loom video of a live **demo of your application** that also describes the use case.
    2. A **written document** addressing each deliverable and answering each question
    3. All relevant code
2. A public (or otherwise shared) link to the **final version of your public application** on Hugging Face (or other)
3. A public link to your **fine-tuned embedding model** on Hugging Face