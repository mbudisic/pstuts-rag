import uuid
import pandas as pd
import numpy as np
from functools import partial
from ragas import EvaluationDataset
from langchain_core.runnables import Runnable
from typing import Set, Tuple, Dict, List
from sklearn.model_selection import train_test_split
import bidict
from typing import Mapping
from itertools import chain
import asyncio
from typing import Optional
from tqdm import tqdm
from .datastore import batch
import numpy as np
from scipy.stats import norm


async def apply_rag_chain_inplace(
    rag_chain: Runnable,
    ragas_ds: EvaluationDataset,
    batch_size: int = 10,
) -> None:
    """
    Apply RAG chain to dataset items in parallel batches.

    Args:
        rag_chain: The RAG chain to apply
        ragas_ds: The dataset to process
        batch_size: Number of items to process in each batch
    """

    async def process_item(item):
        response = await rag_chain.ainvoke({"question": item.user_input})
        item.response = response.content
        item.retrieved_contexts = [
            context.page_content
            for context in response.additional_kwargs["context"]
        ]

    # Process items in batches using the batch function
    for batch_items in batch(list(ragas_ds), size=batch_size):
        tasks = [process_item(item) for item in batch_items]
        await asyncio.gather(*tasks)


def encode_corpus(
    queries: Mapping[str, str],
    corpus: Mapping[str, str],
    input: EvaluationDataset | pd.DataFrame,
) -> Tuple[Mapping[str, str], Mapping[str, str], Dict[str, Set[str]]]:
    """
    Encodes a corpus for information retrieval evaluation.

    Args:
        input: Dataset containing queries and relevant documents
        queries: Mapping of query IDs to query strings
        corpus: Mapping of document IDs to document text

    Returns:
        Tuple containing:
        - Dictionary mapping query IDs to lists of relevant document IDs
        - Bidirectional mapping of query IDs to query strings
        - Bidirectional mapping of document IDs to document text

    https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#informationretrievalevaluator
    """

    queries = bidict.bidict(queries)
    corpus = bidict.bidict(corpus)
    relevant_docs: Dict[str, Set[str]] = {}

    if isinstance(input, EvaluationDataset):
        input = input.to_pandas()

    duplicate_handler = bidict.OnDup(key=bidict.DROP_NEW, val=bidict.DROP_NEW)
    for q, cs in zip(input["user_input"], input["reference_contexts"]):
        queries.put(f"Q_{str(uuid.uuid4())}", q, on_dup=duplicate_handler)
        for c in cs:
            corpus.put(f"D_{str(uuid.uuid4())}", c, on_dup=duplicate_handler)

        q_id = queries.inverse[q]
        c_ids = [corpus.inverse[c] for c in cs]

        relevant_docs[q_id] = set(c_ids)

    # all relevant_docs keys should be in queries and values in corpus
    assert len(relevant_docs.keys() - queries.keys()) == 0
    assert len(set().union(*relevant_docs.values()) - corpus.keys()) == 0

    return dict(queries), dict(corpus), relevant_docs


def train_val_test_split(
    df: pd.DataFrame, ratios: Tuple[int, int, int], seed: int = 42
):

    rlist: list[float] = list(ratios)
    rlist = [float(r) / sum(rlist) for r in rlist]

    print(rlist)

    # assume `df` is your full DataFrame
    # first, split off the test set
    train_val, test = train_test_split(
        df,
        test_size=rlist[2],
        random_state=seed,  # for reproducibility
        shuffle=True,  # default; set to False if you need sequential split
    )
    # then split the remaining 80% into train (60%) and validate (20%)
    train, validate = train_test_split(
        train_val,
        test_size=rlist[1]
        / sum(rlist[0:2]),  # recalculate percentage of the remainder
        random_state=seed,
        shuffle=True,
    )

    final_lengths = [len(x) for x in (train, validate, test)]
    final_ratios = [
        round(float(x) / sum(final_lengths), 2) for x in final_lengths
    ]
    print(
        f"Dataset of {len(df)} split into {final_lengths} which is {final_ratios}"
    )

    return train, validate, test


def summary_stats(df: pd.DataFrame, mean_label="Mean", std_label="StdDev"):
    """
    Adds summary rows (mean and standard deviation) to a pandas DataFrame.

    This function calculates the mean of numeric columns and estimates the standard
    deviation using a jackknife resampling method. It then appends these summary
    statistics as new rows to the original DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame to summarize
        mean_label (str, optional): Label for the mean row. Defaults to "Mean".
        std_label (str, optional): Label for the standard deviation row. Defaults to "StdDev".

    Returns:
        pd.DataFrame: Original DataFrame with summary rows appended

    Raises:
        ValueError: If the DataFrame has fewer than 2 rows (needed for jackknife estimation)
    """
    # 1. Identify numeric vs non-numeric columns
    df = df.infer_objects()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df.columns.difference(numeric_cols)

    n = len(df)
    if n < 2:
        raise ValueError("Need at least two rows for jackknife estimate")

    # 2. Compute the plain mean
    means = df[numeric_cols].mean()

    # 3. Compute jackknife leave-one-out std dev
    jackknife_std = {}
    # total sums up front for speed
    col_sums = df[numeric_cols].sum()
    for col in numeric_cols:
        x = df[col].values
        # leave-one-out means: (sum(x) - x[i])/(n-1)
        loo_means = (col_sums[col] - x) / (n - 1)
        # jackknife variance formula: (n-1)/n * Σ (θ_i – θ̄)²
        theta = loo_means.mean()
        var = (n - 1) / n * np.sum((loo_means - theta) ** 2)
        jackknife_std[col] = np.sqrt(var)

    jackknife_std = pd.Series(jackknife_std)

    # 4. Build a 2×cols DataFrame for the summary rows
    summary = pd.DataFrame(index=[mean_label, std_label], columns=df.columns)

    # fill numeric summaries
    for col in numeric_cols:
        summary.at[mean_label, col] = means[col]
        summary.at[std_label, col] = jackknife_std[col]
    # fill non-numeric with labels
    for col in non_numeric_cols:
        summary.at[mean_label, col] = mean_label
        summary.at[std_label, col] = std_label

    # 5. Concatenate and return
    retval = pd.concat([df, summary], axis=0)
    # Convert columns to numeric where possible, ignoring errors
    retval = retval.apply(partial(pd.to_numeric, **{"errors": "ignore"}))
    # Infer the best data types for each column
    retval = retval.infer_objects()
    return retval


def combine_stats(
    dfs: Tuple[pd.DataFrame, ...], field: str, names: Tuple[str, ...]
) -> pd.DataFrame:
    """
    Combine statistics from multiple DataFrames for a specific field.

    Args:
        dfs: Tuple containing pandas DataFrames to compare
        field: The field name to extract from all DataFrames
        names: Tuple of strings to use as labels for the DataFrames in the output
             (must have the same length as dfs)

    Returns:
        pd.DataFrame: Combined statistics with rows labeled according to names
    """
    if len(dfs) != len(names):
        raise ValueError("Number of DataFrames must match number of names")

    # Extract the field row from each DataFrame and combine them
    field_data = []
    for df in dfs:
        # Get the row for the specified field
        row_data = df.loc[[field]]
        field_data.append(row_data)

    # Combine all the data
    combined_stats = pd.concat(field_data)
    # Set the index to the model names
    combined_stats.index = names
    return combined_stats.select_dtypes(include="number")


def z_test(mu1, mu2, se1, se2):
    """
    Perform a two-sample z-test to compare two means.

    Args:
        mu1: Mean of the first sample
        mu2: Mean of the second sample
        se1: Standard error of the first sample
        se2: Standard error of the second sample

    Returns:
        Tuple[float, float]: z-statistic and p-value for the test
    """

    z = (mu2 - mu1) / np.sqrt(se1**2 + se2**2)
    p = 2 * norm.sf(abs(z))
    return z, p
