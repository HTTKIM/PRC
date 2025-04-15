import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import time
import sys
from dotenv import load_dotenv
load_dotenv()

# 벡터 DB + 임베딩
embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")
vectorstore = FAISS.load_local(
    folder_path=r"C:\\python-proc\\1_DATA_Embedding\\FAISS_store_0414",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

retriever_sim = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
retriever_mmr = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 1.0})

# 유사도 계산 함수
def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def is_semantically_similar(term1, term2, threshold=0.6):
    try:
        vec1 = embedding_model.embed_query(term1)
        vec2 = embedding_model.embed_query(term2)
        return cosine_sim(vec1, vec2) >= threshold
    except:
        return False

# GPT 호출
def stream_gpt_response(prompt):
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3, streaming=True)
    full_text = ""
    for chunk in llm.stream(prompt):
        content = chunk.content or ""
        full_text += content
        yield content
    return full_text

# Streamlit UI
st.set_page_config(layout="wide")
st.title("🧪 GPT 기반 표준 공정 생성기")

with st.form("query_form"):
    col1, col2 = st.columns(2)
    with col1:
        원료 = st.text_input("원료", value="원지")
    with col2:
        생산품 = st.text_input("생산품", value="신문")

    submitted = st.form_submit_button("공정 생성하기")

if submitted:
    query = f"원료: {원료} / 생산품: {생산품}"

    # 검색
    with st.spinner("🔎 유사 공정 검색 중..."):
        with ThreadPoolExecutor() as executor:
            sim_docs = executor.submit(retriever_sim.invoke, query).result()
            mmr_docs = executor.submit(retriever_mmr.invoke, query).result()
        all_docs = {doc.page_content: doc for doc in sim_docs + mmr_docs}.values()

    # 의미 유사도 필터링
    유사공정 = []
    for doc in all_docs:
        if is_semantically_similar(원료, doc.metadata.get("원료", "")) and \
           is_semantically_similar(생산품, doc.metadata.get("생산품", "")):
            유사공정.append(doc.metadata.get("공정", ""))

    if not 유사공정:
        st.warning("❗ 유사한 공정을 찾을 수 없어 GPT 생성이 생략됩니다.")
        st.write("표준 공정을 도출할 수 없습니다.")
    else:
        예시들 = "\n".join([f"- {flow}" for flow in 유사공정])
        system_prompt = (
            "너는 공장을 점검하고 해당 사업장의 공정을 단위공정 기준으로 분석하는 전문가야. "
            "다음 예시는 실제 유사한 사례들이고, 이 중에서 가장 적절한 흐름을 참고하여 공정을 만들어야 해. "
            "거짓말은 허용되지 않으며, 예시와 전혀 관련 없는 내용을 만들어내면 안 돼. "
            "단위공정 명칭은 아래 유사공정에 등장하는 단어를 사용. "
            "답변은 단위공정이 화살표로 연결된 흐름의 형태로 회신해줘."
        )
        final_prompt = f"{system_prompt}\n\n참고 공정 예시:\n{예시들}\n\n표준 공정 작성 대상:\n{query}"

        st.subheader("GPT가 작성한 표준 공정")

        with st.spinner("답변 작성중..."):
            response_box = st.empty()
            stream_text = ""

            for chunk in stream_gpt_response(final_prompt):
                stream_text += chunk
                formatted_text = stream_text.replace("→", " → ")
                response_box.markdown(
                    f"""
                    <div style='font-size:20px; line-height:1.8; font-family: "Noto Sans KR", sans-serif;'>
                        {formatted_text}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # 유사도 평가
        gpt_vector = embedding_model.embed_query(stream_text)
        best_score = 0
        best_match = ""
        for ref in 유사공정:
            ref_vector = embedding_model.embed_query(ref)
            score = cosine_sim(gpt_vector, ref_vector)
            if score > best_score:
                best_score = score
                best_match = ref

        with st.expander("📎 가장 유사한 참고 공정"):
            st.markdown(f"**공정:** {best_match}")
            st.markdown(f"**유사도 점수:** {round(best_score, 4)}")
