import json
import math
import re
from collections import Counter
from pathlib import Path

import pymupdf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = PROJECT_ROOT / "data" / "main.pdf"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "pages.jsonl"

# 第1页是封面，第2～3页是目录，不进入知识库
SKIP_PAGES = {1, 2, 3}


def normalize_lines(text: str) -> list[str]:
    """统一空格格式，并删除空行。"""

    text = text.replace("\u3000", " ")
    text = text.replace("\xa0", " ")

    normalized_lines = []

    for raw_line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()

        if line:
            normalized_lines.append(line)

    return normalized_lines


def extract_raw_pages(pdf_path: Path) -> list[dict]:
    """从PDF逐页提取原始文本行。"""

    pages = []

    with pymupdf.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            text = page.get_text("text", sort=True)
            lines = normalize_lines(text)

            pages.append(
                {
                    "page_number": page_index + 1,
                    "lines": lines,
                }
            )

    return pages


def find_repeated_short_lines(pages: list[dict]) -> set[str]:
    """找出大量页面重复出现的短行，通常是页眉或页脚。"""

    line_counter = Counter()

    for page in pages:
        # 使用set，保证一行在同一页中最多只统计一次
        unique_lines = set(page["lines"])
        line_counter.update(unique_lines)

    # 至少出现在30%的页面中，才认为是重复页眉或页脚
    threshold = max(3, math.ceil(len(pages) * 0.3))

    repeated_lines = {
        line
        for line, count in line_counter.items()
        if count >= threshold and len(line) <= 80
    }

    return repeated_lines

def looks_like_heading(line: str) -> bool:
    """判断一行是否可能是章节标题、摘要标题或图表标题。"""

    if len(line) > 80:
        return False

    heading_patterns = (
        r"^\d+(?:\.\d+)*\s+\S+",
        r"^(摘要|Abstract|关键词|Key\s*words)$",
        r"^(参考文献|致谢|附录)$",
        r"^[图表]\s*\d+(?:-\d+)?",
    )

    return any(
        re.match(pattern, line)
        for pattern in heading_patterns
    )


def merge_wrapped_lines(lines: list[str]) -> str:
    """合并由PDF视觉排版造成的句内换行。"""

    if not lines:
        return ""

    merged_lines = []
    current_text = lines[0]

    for next_line in lines[1:]:
        # 修复英文单词在行末被连字符拆开的情况：
        # experi-\nments -> experiments
        if (
            current_text.endswith("-")
            and next_line
            and next_line[0].islower()
        ):
            current_text = current_text[:-1] + next_line
            continue

        # 遇到章节标题、图表标题时保留结构边界
        if (
            looks_like_heading(current_text)
            or looks_like_heading(next_line)
        ):
            merged_lines.append(current_text)
            current_text = next_line
            continue

        # 完整句子结束后保留换行
        if current_text.endswith(("。", "！", "？", "；")):
            merged_lines.append(current_text)
            current_text = next_line
            continue

        # 中文句子在视觉行末被拆开，直接连接
        if (
            re.search(r"[\u4e00-\u9fff，、：]$", current_text)
            and re.match(r"[\u4e00-\u9fffA-Za-z0-9]", next_line)
        ):
            current_text += next_line
            continue

        # 英文句子在视觉行末被拆开，使用一个空格连接
        if (
            re.search(r"[A-Za-z0-9,]$", current_text)
            and re.match(r"[A-Za-z0-9]", next_line)
        ):
            current_text += " " + next_line
            continue

        # 无法可靠判断时保留换行，避免过度清洗
        merged_lines.append(current_text)
        current_text = next_line

    merged_lines.append(current_text)

    merged_text = "\n".join(merged_lines)

    # 继续清除两个中文字符之间遗留的普通空格
    merged_text = re.sub(
        r"(?<=[\u4e00-\u9fff]) +(?=[\u4e00-\u9fff])",
        "",
        merged_text,
    )

    return merged_text.strip()



def clean_page(
    lines: list[str],
    repeated_lines: set[str],
) -> str:
    """删除重复页眉、页脚和独立页码。"""

    cleaned_lines = []

    for line in lines:
        # 删除在大量页面重复出现的页眉或页脚
        if line in repeated_lines:
            continue

        # 删除“1”“第 1 页”“1 页”等独立页码
        if re.fullmatch(r"(第\s*)?\d+\s*(页)?", line):
            continue

        cleaned_lines.append(line)

    return merge_wrapped_lines(cleaned_lines)





def save_cleaned_pages(
    pages: list[dict],
    repeated_lines: set[str],
) -> list[dict]:
    """清洗页面并保存为JSONL文件。"""

    cleaned_pages = []

    for page in pages:
        page_number = page["page_number"]

        if page_number in SKIP_PAGES:
            continue

        cleaned_text = clean_page(
            lines=page["lines"],
            repeated_lines=repeated_lines,
        )

        if not cleaned_text:
            continue

        cleaned_pages.append(
            {
                "page_number": page_number,
                "text": cleaned_text,
                "character_count": len(cleaned_text),
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for page in cleaned_pages:
            json_line = json.dumps(page, ensure_ascii=False)
            output_file.write(json_line + "\n")

    return cleaned_pages


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"没有找到PDF：{PDF_PATH}")

    raw_pages = extract_raw_pages(PDF_PATH)
    repeated_lines = find_repeated_short_lines(raw_pages)
    cleaned_pages = save_cleaned_pages(raw_pages, repeated_lines)

    print(f"PDF原始页数：{len(raw_pages)}")
    print(f"跳过页面：{sorted(SKIP_PAGES)}")
    print(f"保留页面数：{len(cleaned_pages)}")
    print(f"识别到的重复短行：{len(repeated_lines)}")

    print("\n被识别为页眉或页脚的内容：")

    for line in sorted(repeated_lines):
        print(f"- {line}")

    print("\n清洗后的前3页：")

    for page in cleaned_pages[:3]:
        preview = page["text"][:300].replace("\n", " ")

        print("-" * 60)
        print(f"PDF页码：{page['page_number']}")
        print(f"字符数：{page['character_count']}")
        print(f"内容预览：{preview}")

    print(f"\n输出文件：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()