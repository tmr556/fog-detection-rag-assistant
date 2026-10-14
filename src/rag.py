import argparse
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from ollama import Client

try:
    from .search import search
except ImportError:
    from search import search


PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")


SYSTEM_PROMPT = """
你是一个雾天目标检测课程报告问答助手。

请严格遵守以下规则：

1. 只能根据用户提供的参考资料回答，不得使用参考资料之外的知识补充实验结论。
2. 回答中的实验数据、性能比较、原因分析和模型结论必须能在参考资料中找到依据。
3. 引用依据时使用“[PDF第X页]”格式，X必须是参考资料中真实提供的页码。
4. 不得虚构不存在的页码、实验指标、模型结构或实验现象。
5. 如果参考资料不足以回答问题，明确回答：
   “根据当前检索到的报告内容，无法确定该问题的答案。”
6. 使用简洁、准确的中文回答，先给出结论，再解释依据。
7. 回答控制在三个要点以内，不要重复总结，不要单独列出“依据页码”。
8. 引用只能使用“[PDF第X页]”格式，不得在引用中输出文本块编号。
9. 多个页码应分别标注，例如“[PDF第51页][PDF第52页]”。
10. 不要复述用户问题、系统规则、提示词或检索过程。
""".strip()


def get_config() -> dict:
    """从.env中读取RAG运行配置。"""

    config = {
        "base_url": os.getenv(
            "LLM_BASE_URL",
            "http://localhost:11434/v1",
        ),
        "api_key": os.getenv(
            "LLM_API_KEY",
            "ollama",
        ),
        "model": os.getenv(
            "LLM_MODEL",
            "qwen3:4b",
        ),
        "top_k": int(
            os.getenv("TOP_K", "5")
        ),
        "similarity_threshold": float(
            os.getenv(
                "SIMILARITY_THRESHOLD",
                "0.535",
            )
        ),
    }

    return config


def build_context(results: list[dict]) -> str:
    """把检索结果整理成大模型能够阅读的参考资料。"""

    context_parts = []

    for result in results:
        context_parts.append(
            "\n".join(
                [
                    (
                        f"【参考资料{result['rank']}｜"
                        f"PDF第{result['page_number']}页｜"
                        
                    ),
                    result["text"],
                ]
            )
        )

    return "\n\n".join(context_parts)


def sanitize_citations(
    answer: str,
    results: list[dict],
) -> tuple[str, list[int]]:
    """移除不属于本次检索结果的PDF页码引用。"""

    valid_pages = {
        int(result["page_number"])
        for result in results
    }
    invalid_pages = set()
    citation_pattern = re.compile(
        r"\[PDF第(\d+)页\]"
    )

    def replace_citation(match: re.Match) -> str:
        page_number = int(match.group(1))

        if page_number in valid_pages:
            return match.group(0)

        invalid_pages.add(page_number)
        return ""

    sanitized_answer = citation_pattern.sub(
        replace_citation,
        answer,
    )

    return sanitized_answer.strip(), sorted(invalid_pages)


def generate_answer(
    question: str,
    context: str,
    config: dict,
) -> str:
    """调用本地Qwen3，根据检索内容生成答案。"""

    client = Client(
        host=config["base_url"],
    )

    user_prompt = f"""
用户问题：
{question}

参考资料：
{context}

请根据参考资料回答用户问题，并标注PDF页码。
""".strip()

    response = client.chat(
        model=config["model"],
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        think=False,
        stream=False,
        options={
            "temperature": 0.2,
            "num_predict": 800,
        },
    )

    answer = response.message.content

    if not answer or not answer.strip():
        raise RuntimeError(
            "模型请求成功，但没有返回最终回答内容"
        )

    return answer.strip()


def print_sources(results: list[dict]) -> None:
    """打印本次回答使用的检索结果，方便检查。"""

    print("\n检索来源")
    print("=" * 70)

    for result in results:
        print(
            f"- 排名{result['rank']} | "
            f"{result['chunk_id']} | "
            f"PDF第{result['page_number']}页 | "
            f"相似度={result['similarity']:.4f}"
        )


def run_rag(question: str) -> dict:
    """执行RAG问答并返回供命令行或网页使用的结构化结果。"""

    question = question.strip()

    if not question:
        raise ValueError("问题不能为空")

    config = get_config()
    results = search(
        query=question,
        top_k=config["top_k"],
    )

    common_result = {
        "question": question,
        "model": config["model"],
        "top_k": config["top_k"],
        "threshold": config["similarity_threshold"],
        "results": results,
    }

    if not results:
        return {
            **common_result,
            "status": "empty",
            "top_similarity": None,
            "answer": "知识库没有返回任何检索结果。",
            "invalid_citation_pages": [],
        }

    top_similarity = max(
        result["similarity"]
        for result in results
    )

    if top_similarity < config["similarity_threshold"]:
        return {
            **common_result,
            "status": "rejected",
            "top_similarity": top_similarity,
            "answer": (
                "这个问题超出了当前课程报告的知识范围，"
                "我无法根据报告内容作答。"
            ),
            "invalid_citation_pages": [],
        }

    context = build_context(results)

    try:
        generated_answer = generate_answer(
            question=question,
            context=context,
            config=config,
        )
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError(
            "无法连接Ollama。请确认Ollama正在运行，"
            f"并且已经下载模型{config['model']}。"
        ) from error

    answer, invalid_citation_pages = sanitize_citations(
        answer=generated_answer,
        results=results,
    )

    return {
        **common_result,
        "status": "answered",
        "top_similarity": top_similarity,
        "answer": answer,
        "invalid_citation_pages": invalid_citation_pages,
    }


def answer_question(question: str) -> None:
    """执行一次完整的RAG问答并输出到终端。"""

    print(f"\n问题：{question}")

    result = run_rag(question)

    print(f"检索数量：Top {result['top_k']}")
    print(f"拒答阈值：{result['threshold']:.4f}")

    if result["top_similarity"] is not None:
        print(
            "最高相似度："
            f"{result['top_similarity']:.4f}"
        )

    if result["status"] == "answered":
        print(f"使用本地模型：{result['model']}")

    print("\n回答")
    print("=" * 70)
    print(result["answer"])

    if result["invalid_citation_pages"]:
        invalid_pages = ", ".join(
            str(page)
            for page in result["invalid_citation_pages"]
        )
        print(
            "\n引用校验：已移除不在检索结果中的页码："
            f"{invalid_pages}"
        )

    print_sources(result["results"])


def parse_arguments():
    """读取命令行问题。"""

    parser = argparse.ArgumentParser(
        description="雾天目标检测报告RAG问答"
    )

    parser.add_argument(
        "question",
        nargs="?",
        help="需要向报告提问的问题",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    question = arguments.question

    if not question:
        question = input(
            "请输入关于课程报告的问题："
        ).strip()

    if not question:
        raise ValueError("问题不能为空")

    answer_question(question)


if __name__ == "__main__":
    main()
