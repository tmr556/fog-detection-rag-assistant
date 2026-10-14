import argparse
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_PATH = PROJECT_ROOT / "chroma_db"

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
COLLECTION_NAME = "fog_detection_report"

QUERY_INSTRUCTION = (
    "为这个句子生成表示以用于检索相关文章："
)


def load_model() -> SentenceTransformer:
    """加载与文档索引相同的Embedding模型。"""

    print(f"正在加载模型：{MODEL_NAME}")

    return SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )


def load_collection():
    """连接本地ChromaDB并读取报告集合。"""

    if not CHROMA_PATH.exists():
        raise FileNotFoundError(
            "没有找到ChromaDB，请先运行build_index.py"
        )

    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
    )

    return collection


def search(
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """根据自然语言问题检索最相关的文本块。"""

    if not query.strip():
        raise ValueError("查询问题不能为空")

    model = load_model()
    collection = load_collection()

    instructed_query = QUERY_INSTRUCTION + query.strip()

    query_embedding = model.encode(
        instructed_query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    result = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    retrieved_chunks = []

    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    for rank, (
        chunk_id,
        document,
        metadata,
        distance,
    ) in enumerate(
        zip(
            ids,
            documents,
            metadatas,
            distances,
        ),
        start=1,
    ):
        retrieved_chunks.append(
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "page_number": metadata["page_number"],
                "distance": distance,
                "similarity": 1 - distance,
                "text": document,
            }
        )

    return retrieved_chunks


def print_results(
    query: str,
    results: list[dict],
) -> None:
    """以便于人工检查的格式打印结果。"""

    print("\n" + "=" * 70)
    print(f"查询问题：{query}")
    print(f"返回结果：{len(results)} 个")
    print("=" * 70)

    for result in results:
        print("\n" + "-" * 70)
        print(f"排名：{result['rank']}")
        print(f"文本块：{result['chunk_id']}")
        print(f"PDF页码：{result['page_number']}")
        print(f"余弦距离：{result['distance']:.4f}")
        print(f"相似度：{result['similarity']:.4f}")
        print("文本内容：")
        print(result["text"])


def parse_arguments():
    """读取命令行中的问题和Top-k参数。"""

    parser = argparse.ArgumentParser(
        description="检索雾天目标检测报告中的相关内容"
    )

    parser.add_argument(
        "query",
        help="需要检索的自然语言问题",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="返回的文本块数量，默认为5",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    results = search(
        query=arguments.query,
        top_k=arguments.top_k,
    )

    print_results(
        query=arguments.query,
        results=results,
    )


if __name__ == "__main__":
    main()