import json
from pathlib import Path
from statistics import mean

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

QUESTIONS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "retrieval_questions.jsonl"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "retrieval_results.json"
)

CHROMA_PATH = PROJECT_ROOT / "chroma_db"

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
COLLECTION_NAME = "fog_detection_report"

QUERY_INSTRUCTION = (
    "为这个句子生成表示以用于检索相关文章："
)

TOP_K = 5


def load_questions(path: Path) -> list[dict]:
    """读取并验证人工评测问题。"""

    questions = []
    required_fields = {
        "id",
        "question",
        "answerable",
        "expected_pages",
        "category",
    }

    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(
            input_file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"评测文件第{line_number}行格式错误："
                    f"{error}"
                ) from error

            missing_fields = required_fields - item.keys()

            if missing_fields:
                raise ValueError(
                    f"第{line_number}行缺少字段："
                    f"{sorted(missing_fields)}"
                )

            if (
                item["answerable"]
                and not item["expected_pages"]
            ):
                raise ValueError(
                    f"{item['id']}是可回答问题，"
                    "但没有标注相关页码"
                )

            questions.append(item)

    if not questions:
        raise ValueError("评测集中没有问题")

    return questions


def load_model() -> SentenceTransformer:
    """加载与索引构建阶段相同的Embedding模型。"""

    print(f"正在加载模型：{MODEL_NAME}")

    return SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )


def load_collection():
    """连接已经构建好的ChromaDB集合。"""

    if not CHROMA_PATH.exists():
        raise FileNotFoundError(
            "没有找到chroma_db，"
            "请先运行src/build_index.py"
        )

    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
    )


def batch_retrieve(
    questions: list[dict],
    model: SentenceTransformer,
    collection,
) -> list[dict]:
    """批量生成问题向量并执行Top-k检索。"""

    instructed_queries = [
        QUERY_INSTRUCTION + item["question"]
        for item in questions
    ]

    query_embeddings = model.encode(
        instructed_queries,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    raw_results = collection.query(
        query_embeddings=query_embeddings.tolist(),
        n_results=TOP_K,
        include=[
            "metadatas",
            "distances",
        ],
    )

    details = []

    for question_index, question in enumerate(questions):
        chunk_ids = raw_results["ids"][question_index]
        metadatas = raw_results["metadatas"][
            question_index
        ]
        distances = raw_results["distances"][
            question_index
        ]

        retrieved_pages = [
            metadata["page_number"]
            for metadata in metadatas
        ]

        similarities = [
            1 - distance
            for distance in distances
        ]

        relevant_ranks = [
            rank
            for rank, page_number in enumerate(
                retrieved_pages,
                start=1,
            )
            if page_number in question["expected_pages"]
        ]

        first_relevant_rank = (
            min(relevant_ranks)
            if relevant_ranks
            else None
        )

        details.append(
            {
                "id": question["id"],
                "question": question["question"],
                "category": question["category"],
                "answerable": question["answerable"],
                "expected_pages": question[
                    "expected_pages"
                ],
                "retrieved_pages": retrieved_pages,
                "retrieved_chunk_ids": chunk_ids,
                "similarities": [
                    round(score, 4)
                    for score in similarities
                ],
                "top_similarity": round(
                    similarities[0],
                    4,
                ),
                "first_relevant_rank": (
                    first_relevant_rank
                ),
            }
        )

    return details


def calculate_metrics(
    details: list[dict],
) -> dict:
    """计算可回答问题的Hit@k和MRR@5。"""

    answerable_results = [
        item
        for item in details
        if item["answerable"]
    ]

    if not answerable_results:
        raise ValueError("没有可回答问题用于计算指标")

    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0
    reciprocal_ranks = []

    for item in answerable_results:
        rank = item["first_relevant_rank"]

        if rank is None:
            reciprocal_ranks.append(0.0)
            continue

        if rank <= 1:
            hit_at_1 += 1

        if rank <= 3:
            hit_at_3 += 1

        if rank <= 5:
            hit_at_5 += 1

        reciprocal_ranks.append(1 / rank)

    total = len(answerable_results)

    return {
        "answerable_questions": total,
        "hit_at_1": round(hit_at_1 / total, 4),
        "hit_at_3": round(hit_at_3 / total, 4),
        "hit_at_5": round(hit_at_5 / total, 4),
        "mrr_at_5": round(
            mean(reciprocal_ranks),
            4,
        ),
    }


def analyze_answerability(
    details: list[dict],
) -> dict:
    """比较可回答与不可回答问题的最高相似度。"""

    positive_scores = [
        item["top_similarity"]
        for item in details
        if item["answerable"]
    ]

    negative_scores = [
        item["top_similarity"]
        for item in details
        if not item["answerable"]
    ]

    analysis = {
        "positive_min": round(min(positive_scores), 4),
        "positive_mean": round(mean(positive_scores), 4),
        "positive_max": round(max(positive_scores), 4),
        "negative_min": round(min(negative_scores), 4),
        "negative_mean": round(mean(negative_scores), 4),
        "negative_max": round(max(negative_scores), 4),
        "suggested_threshold": None,
    }

    # 只有正负样本完全分离时，才给出候选阈值。
    if min(positive_scores) > max(negative_scores):
        analysis["suggested_threshold"] = round(
            (
                min(positive_scores)
                + max(negative_scores)
            )
            / 2,
            4,
        )

    return analysis


def print_report(
    details: list[dict],
    metrics: dict,
    answerability: dict,
) -> None:
    """在终端打印便于检查的评测报告。"""

    print("\n逐题结果")
    print("=" * 80)

    for item in details:
        if item["answerable"]:
            rank = item["first_relevant_rank"]

            status = (
                f"HIT@{rank}"
                if rank is not None
                else "MISS"
            )
        else:
            status = "OUT-OF-DOMAIN"

        print(
            f"{item['id']:>3} | "
            f"{status:<13} | "
            f"top={item['top_similarity']:.4f} | "
            f"pages={item['retrieved_pages']}"
        )

    print("\n检索指标")
    print("=" * 80)
    print(
        f"Hit@1：{metrics['hit_at_1']:.4f}"
    )
    print(
        f"Hit@3：{metrics['hit_at_3']:.4f}"
    )
    print(
        f"Hit@5：{metrics['hit_at_5']:.4f}"
    )
    print(
        f"MRR@5：{metrics['mrr_at_5']:.4f}"
    )

    print("\n相似度分布")
    print("=" * 80)
    print(
        "可回答问题："
        f"min={answerability['positive_min']:.4f}, "
        f"mean={answerability['positive_mean']:.4f}, "
        f"max={answerability['positive_max']:.4f}"
    )
    print(
        "不可回答问题："
        f"min={answerability['negative_min']:.4f}, "
        f"mean={answerability['negative_mean']:.4f}, "
        f"max={answerability['negative_max']:.4f}"
    )

    threshold = answerability["suggested_threshold"]

    if threshold is None:
        print("正负样本存在重叠，暂不建议设置固定阈值。")
    else:
        print(f"候选拒答阈值：{threshold:.4f}")


def save_results(
    details: list[dict],
    metrics: dict,
    answerability: dict,
) -> None:
    """将完整评测结果保存为JSON。"""

    result = {
        "model": MODEL_NAME,
        "collection": COLLECTION_NAME,
        "top_k": TOP_K,
        "metrics": metrics,
        "answerability_analysis": answerability,
        "details": details,
    }

    with RESULTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            result,
            output_file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n完整结果已保存：{RESULTS_PATH}")


def main() -> None:
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(
            f"没有找到评测集：{QUESTIONS_PATH}"
        )

    questions = load_questions(QUESTIONS_PATH)

    print(f"读取评测问题：{len(questions)} 条")

    model = load_model()
    collection = load_collection()

    details = batch_retrieve(
        questions=questions,
        model=model,
        collection=collection,
    )

    metrics = calculate_metrics(details)
    answerability = analyze_answerability(details)

    print_report(
        details=details,
        metrics=metrics,
        answerability=answerability,
    )

    save_results(
        details=details,
        metrics=metrics,
        answerability=answerability,
    )


if __name__ == "__main__":
    main()