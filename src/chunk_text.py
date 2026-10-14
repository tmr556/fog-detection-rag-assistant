import json
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "pages.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"

CHUNK_SIZE = 450
CHUNK_OVERLAP = 80

BREAK_CHARACTERS = ("。", "！", "？", "；", "\n")


def load_pages(input_path: Path) -> list[dict]:
    """读取页面级JSONL数据。"""

    pages = []

    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue

            try:
                page = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"第{line_number}行不是有效JSON：{error}"
                ) from error

            pages.append(page)

    return pages


def find_chunk_end(
    text: str,
    start: int,
    target_end: int,
) -> int:
    """在目标长度附近寻找更自然的句子边界。"""

    if target_end >= len(text):
        return len(text)

    # 只在文本块后40%的范围内寻找结束标点，
    # 避免为了寻找标点把文本块切得太短。
    search_start = start + int(CHUNK_SIZE * 0.6)
    candidate = text[search_start:target_end]

    boundary_positions = [
        candidate.rfind(character)
        for character in BREAK_CHARACTERS
    ]

    best_position = max(boundary_positions)

    if best_position == -1:
        return target_end

    return search_start + best_position + 1


def split_page(page: dict) -> list[dict]:
    """把单页文本切分成若干有重叠的文本块。"""

    page_number = page["page_number"]
    text = page["text"]

    chunks = []
    start = 0
    chunk_index = 1

    while start < len(text):
        target_end = min(start + CHUNK_SIZE, len(text))

        end = find_chunk_end(
            text=text,
            start=start,
            target_end=target_end,
        )

        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(
                {
                    "chunk_id": (
                        f"p{page_number:04d}_"
                        f"c{chunk_index:03d}"
                    ),
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "character_count": len(chunk_text),
                    "char_start": start,
                    "char_end": end,
                    "source": "main.pdf",
                }
            )

            chunk_index += 1

        if end >= len(text):
            break

        # 下一个文本块向前回退80字符，形成重叠区域
        next_start = end - CHUNK_OVERLAP

        # 防止异常情况下循环无法前进
        start = max(start + 1, next_start)

    return chunks


def save_chunks(chunks: list[dict]) -> None:
    """将文本块保存为JSONL。"""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for chunk in chunks:
            json_line = json.dumps(
                chunk,
                ensure_ascii=False,
            )
            output_file.write(json_line + "\n")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"没有找到页面数据：{INPUT_PATH}"
        )

    pages = load_pages(INPUT_PATH)

    all_chunks = []

    for page in pages:
        page_chunks = split_page(page)
        all_chunks.extend(page_chunks)

    save_chunks(all_chunks)

    chunk_sizes = [
        chunk["character_count"]
        for chunk in all_chunks
    ]

    print(f"输入页面数：{len(pages)}")
    print(f"生成文本块数：{len(all_chunks)}")
    print(f"最短文本块：{min(chunk_sizes)} 字符")
    print(f"最长文本块：{max(chunk_sizes)} 字符")
    print(f"平均文本块：{mean(chunk_sizes):.1f} 字符")
    print(f"输出文件：{OUTPUT_PATH}")

    print("\n前3个文本块：")

    for chunk in all_chunks[:3]:
        print("-" * 60)
        print(f"文本块ID：{chunk['chunk_id']}")
        print(f"PDF页码：{chunk['page_number']}")
        print(f"字符范围：{chunk['char_start']}～{chunk['char_end']}")
        print(f"字符数：{chunk['character_count']}")
        print(chunk["text"][:300].replace("\n", " "))


if __name__ == "__main__":
    main()