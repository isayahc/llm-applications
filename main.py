import matplotlib.pyplot as plt
from rag.generate import generate_responses
from rag.evaluate import evaluate_responses
from rag.config import EMBEDDING_DIMENSIONS, MAX_CONTEXT_LENGTHS, EFS_DIR
from rag.config import ROOT_DIR
#, EVALUATOR 
# from rag.config import EXPERIMENTS_DIR, REFERENCES_FILE_PATH, NUM_SAMPLES      
import numpy as np
import json
from pathlib import Path
import ray


EVALUATOR = "gpt-4"
EXPERIMENTS_DIR = Path(ROOT_DIR, "experiments")
REFERENCES_FILE_PATH = Path(EXPERIMENTS_DIR, "references", "gpt-4.json")
NUM_SAMPLES = None

DOCS_DIR = Path(EFS_DIR, "docs.ray.io/en/master/")


def get_retrieval_score(references, generated):
    matches = np.zeros(len(references))
    for i in range(len(references)):
        reference_source = references[i]["source"].split("#")[0]
        if not reference_source:
            matches[i] = 1
            continue
        for source in generated[i]["sources"]:
            # sections don't have to perfectly match
            if reference_source == source.split("#")[0]:
                matches[i] = 1
                continue
    retrieval_score = np.mean(matches)
    return retrieval_score


def run_experiment(
    experiment_name,
    chunk_size, chunk_overlap, num_chunks,
    embedding_model_name, llm, evaluator,
    docs_dir, experiments_dir, references_fp,
    num_samples=None):
    """Generate responses and evaluate them."""
    
    # Generate responses
    generation_system_content = "Answer the query using the context provided. Be succinct."
    generate_responses(
        experiment_name=experiment_name, 
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap, 
        num_chunks=num_chunks,
        embedding_model_name=embedding_model_name, 
        llm=llm, 
        temperature=0.0, 
        max_context_length=MAX_CONTEXT_LENGTHS[llm], 
        system_content=generation_system_content,
        assistant_content="",
        docs_dir=docs_dir,
        experiments_dir=experiments_dir,
        references_fp=references_fp,
        num_samples=num_samples)

    # Evaluate responses
    evaluation_system_content = """
        Your job is to rate the quality of our generated answer {generated_answer}
        given a query {query} and a reference answer {reference_answer}.
        Your score has to be between 1 and 5.
        You must return your response in a line with only the score.
        Do not return answers in any other format.
        On a separate line provide your reasoning for the score as well.
        """
    evaluate_responses(
        experiment_name=experiment_name,
        evaluator=evaluator, 
        temperature=0.0, 
        max_context_length=MAX_CONTEXT_LENGTHS[evaluator],
        system_content=evaluation_system_content,
        assistant_content="",
        experiments_dir=experiments_dir,
        references_fp=references_fp,
        responses_fp=str(Path(experiments_dir, "responses", f"{experiment_name}.json")),
        num_samples=num_samples)
    

def print_experiment(experiment_name, experiments_dir, evaluator=EVALUATOR):
    eval_fp = Path(experiments_dir, "evaluations", f"{experiment_name}_{evaluator}.json")
    with open(eval_fp, "r") as fp:
        d = json.load(fp)
    retrieval_score = d["retrieval_score"]
    quality_score = d["quality_score"]
    print (experiment_name)
    print ("  retrieval score:", retrieval_score)
    print ("  quality score:", quality_score)
    print ()
    return {"retrieval_score": retrieval_score, "quality_score": quality_score}


def plot_scores(scores):
    # Prepare data for plotting
    experiment_names = list(scores.keys())
    retrieval_scores = [scores[experiment_name]["retrieval_score"] for experiment_name in experiment_names]
    quality_scores = [scores[experiment_name]["quality_score"] for experiment_name in experiment_names]
    
    # Plotting
    plt.figure(figsize=(10, 3))
    for i, experiment_name in enumerate(experiment_names):
        plt.scatter(quality_scores[i], retrieval_scores[i], label=experiment_name)
        plt.text(quality_scores[i]+0.005, retrieval_scores[i]+0.005, experiment_name, ha="right")
        
    # Add labels and title
    plt.xlabel("Quality Score")
    plt.ylabel("Retrieval Score")
    plt.legend(title="Experiments")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":

    EVALUATOR = "gpt-4"
    EXPERIMENTS_DIR = Path(ROOT_DIR, "experiments")
    REFERENCES_FILE_PATH = Path(EXPERIMENTS_DIR, "references", "gpt-4.json")
    NUM_SAMPLES = None

    DOCS_DIR = Path(EFS_DIR, "docs.ray.io/en/master/")

    ds = ray.data.from_items([{"path": path} for path in DOCS_DIR.rglob("*.html") if not path.is_dir()])

    print(f"{ds.count()} documents")
    llm = "gpt-3.5-turbo"
    num_chunks = 0
    experiment_name = f"without-context"
    run_experiment(
        experiment_name=experiment_name, 
        chunk_size=300, 
        chunk_overlap=50,
        num_chunks=num_chunks,
        embedding_model_name="thenlper/gte-base",
        llm=llm,
        evaluator=EVALUATOR,
        docs_dir=DOCS_DIR, 
        experiments_dir=EXPERIMENTS_DIR, 
        references_fp=REFERENCES_FILE_PATH,
        num_samples=NUM_SAMPLES)

    # With context
    num_chunks = 5
    experiment_name = "with-context"
    run_experiment(
        experiment_name=experiment_name, 
        chunk_size=300, 
        chunk_overlap=50, 
        num_chunks=num_chunks,
        embedding_model_name="thenlper/gte-base",
        llm=llm,
        evaluator=EVALUATOR,
        docs_dir=DOCS_DIR, 
        experiments_dir=EXPERIMENTS_DIR, 
        references_fp=REFERENCES_FILE_PATH,
        num_samples=NUM_SAMPLES)

    scores = {}
    for experiment_name in ["without-context", "with-context"]:
        scores[experiment_name] = print_experiment(experiment_name, EXPERIMENTS_DIR)
    plot_scores(scores=scores)