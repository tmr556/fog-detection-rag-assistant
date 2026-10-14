"""Streamlit entry point for the fog-detection RAG assistant."""

import streamlit as st

from src.rag import run_rag


st.set_page_config(
    page_title="雾天目标检测知识库助手",
    page_icon="🌫️",
    layout="wide",
)

st.title("🌫️ 雾天目标检测知识库助手")
st.caption(
    "基于课程设计报告构建的本地RAG系统，"
    "支持语义检索、阈值拒答与PDF页码溯源。"
)

with st.sidebar:
    st.subheader("系统信息")
    st.markdown(
        """
        - **知识库**：雾天目标检测课程报告
        - **Embedding**：BAAI/bge-small-zh-v1.5
        - **向量数据库**：ChromaDB
        - **生成模型**：Qwen3 4B Instruct（本地）
        - **检索数量**：Top 5
        - **拒答阈值**：0.535
        """
    )
    st.divider()
    st.caption(
        "回答仅基于报告中的检索内容生成，"
        "页码为原PDF页码。"
    )

st.markdown("**第一次使用？可以从这些问题开始：**")

example_questions = (
    "这个雾天检测项目是做什么的？",
    "报告使用了哪些数据集？",
    "V3C为什么没有超过V2？",
)
example_columns = st.columns(len(example_questions))

for index, example_question in enumerate(example_questions):
    if example_columns[index].button(
        example_question,
        use_container_width=True,
    ):
        st.session_state["question_input"] = example_question

with st.form("question_form"):
    question = st.text_area(
        "请输入关于报告的问题",
        placeholder=(
            "例如：V3C为什么没有超过V2？"
        ),
        height=100,
        key="question_input",
    )
    submitted = st.form_submit_button(
        "开始问答",
        type="primary",
        use_container_width=True,
    )

if submitted:
    if not question.strip():
        st.warning("请先输入一个问题。")
    else:
        try:
            with st.spinner("正在检索报告并生成回答……"):
                result = run_rag(question)
        except Exception as error:
            st.error(
                "运行失败。请确认Ollama正在运行，"
                "并已下载配置中的模型。"
            )
            with st.expander("查看错误详情"):
                st.exception(error)
        else:
            metric_columns = st.columns(3)
            metric_columns[0].metric(
                "最高相似度",
                (
                    f"{result['top_similarity']:.4f}"
                    if result["top_similarity"] is not None
                    else "无结果"
                ),
            )
            metric_columns[1].metric(
                "拒答阈值",
                f"{result['threshold']:.4f}",
            )
            metric_columns[2].metric(
                "生成模型",
                result["model"],
            )

            st.subheader("回答")

            if result["status"] == "answered":
                st.markdown(result["answer"])
            elif result["status"] == "rejected":
                st.warning(result["answer"])
            else:
                st.info(result["answer"])

            if result["invalid_citation_pages"]:
                st.caption(
                    "系统已自动移除不属于本次检索结果的"
                    "页码引用。"
                )

            with st.expander(
                "查看检索依据",
                expanded=False,
            ):
                for item in result["results"]:
                    st.markdown(
                        f"**排名 {item['rank']} · "
                        f"PDF第 {item['page_number']} 页 · "
                        f"相似度 `{item['similarity']:.4f}`**"
                    )
                    st.write(item["text"])

                    if item != result["results"][-1]:
                        st.divider()

