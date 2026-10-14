import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks.jsonl"
)

CHROMA_PATH = PROJECT_ROOT / "chroma_db"

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
COLLECTION_NAME = "fog_detection_report"
BATCH_SIZE = 32


def load_chunks(chunks_path: Path) -> list[dict]:
    """读取文本分块数据，并检查必要字段。"""

    chunks = []

    required_fields = {
        "chunk_id",
        "page_number",
        "text",
    }

    with chunks_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"第{line_number}行不是有效JSON：{error}"
                ) from error

            missing_fields = required_fields - chunk.keys()

            if missing_fields:
                raise ValueError(
                    f"第{line_number}行缺少字段："
                    f"{sorted(missing_fields)}"
                )

            if not chunk["text"].strip():
                raise ValueError(
                    f"第{line_number}行的文本为空"
                )

            chunks.append(chunk)

    if not chunks:
        raise ValueError("没有读取到任何文本块")

    return chunks


def load_embedding_model() -> SentenceTransformer:
    """加载本地Embedding模型。"""

    print(f"正在加载Embedding模型：{MODEL_NAME}")
    print("首次运行需要从Hugging Face下载模型文件。")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )

    print(
        "向量维度："
        f"{model.get_embedding_dimension()}"
    )
    print(f"最大序列长度：{model.max_seq_length}")

    return model


def encode_chunks(
    model: SentenceTransformer,
    chunks: list[dict],
):
    """将所有文本块批量转换成归一化向量。"""

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings


def save_to_chroma(
    chunks: list[dict],
    embeddings,
) -> int:
    """将向量、文本和页码保存到ChromaDB。"""

    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": MODEL_NAME,
        },
        embedding_function=None,
    )

    ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    metadatas = [
        {
            "page_number": chunk["page_number"],
            "chunk_index": chunk["chunk_index"],
            "character_count": chunk["character_count"],
            "source": chunk["source"],
        }
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings.tolist(),
    )

    return collection.count()


def main() -> None:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"没有找到文本块文件：{CHUNKS_PATH}"
        )

    chunks = load_chunks(CHUNKS_PATH)

    print(f"读取文本块：{len(chunks)} 个")

    model = load_embedding_model()

    print("\n正在生成文本向量……")

    embeddings = encode_chunks(
        model=model,
        chunks=chunks,
    )

    print(f"向量矩阵形状：{embeddings.shape}")

    stored_count = save_to_chroma(
        chunks=chunks,
        embeddings=embeddings,
    )

    print("\n向量知识库构建完成")
    print(f"ChromaDB位置：{CHROMA_PATH}")
    print(f"集合名称：{COLLECTION_NAME}")
    print(f"数据库记录数：{stored_count}")


if __name__ == "__main__":
    main()