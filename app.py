import os
import requests
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI

# 페이지 기본 설정
st.set_page_config(page_title="나와 닮은 세계 지도자 찾기", page_icon="🌍", layout="centered")

# API 키 불러오기 (Streamlit Secrets 또는 로컬 환경변수)
api_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API 키가 설정되지 않았습니다. Secrets 설정을 확인해 주세요.")
    st.stop()

client = OpenAI(api_key=api_key)

CSV_PATH = "leaders_100.csv"
EMB_PATH = "leader_embeddings.npy"

QUESTIONS = [
    "1. [위기 대처] 예상치 못한 큰 위기나 실패가 닥쳤을 때, 가장 먼저 어떤 생각이 들고 어떻게 행동하시나요?",
    "2. [의사결정] 중요한 결정을 내려야 할 때, 철저한 데이터와 논리를 따르시나요, 아니면 직관과 신념을 따르시나요? 그 이유도 적어주세요.",
    "3. [갈등과 소통] 주변 사람이나 조직과 의견이 크게 충돌할 때, 주로 어떤 방식으로 갈등을 해결하시나요?",
    "4. [동기 부여] 당신을 가장 뜨겁게 움직이게 만드는 삶의 목표나 핵심 가치는 무엇인가요?",
    "5. [일하는 방식] 새로운 프로젝트를 시작할 때, 치밀하게 계획을 세우는 편인가요, 아니면 일단 부딪히며 수정하는 편인가요?"
]

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    embeddings = np.load(EMB_PATH)
    return df, embeddings

@st.cache_data(show_spinner=False)
def get_leader_image(name):
    """위키백과 공식 API를 이용해 인물 대표 이미지 URL을 무료(토큰 0원)로 가져옵니다."""
    headers = {"User-Agent": "WorldLeaderMatchApp/1.0 (contact@example.com)"}
    
    # 한국어 위키백과 -> 실패 시 영어 위키백과 순서로 검색
    for lang in ["ko", "en"]:
        try:
            search_url = f"https://{lang}.wikipedia.org/w/api.php"
            search_params = {
                "action": "query",
                "list": "search",
                "srsearch": name,
                "format": "json",
                "srlimit": 1
            }
            res = requests.get(search_url, params=search_params, headers=headers, timeout=4).json()
            search_results = res.get("query", {}).get("search", [])
            if not search_results:
                continue
            
            title = search_results[0]["title"]
            
            # 해당 문서의 대표 썸네일 이미지 URL 가져오기
            img_params = {
                "action": "query",
                "titles": title,
                "prop": "pageimages",
                "format": "json",
                "pithumbsize": 400
            }
            img_res = requests.get(search_url, params=img_params, headers=headers, timeout=4).json()
            pages = img_res.get("query", {}).get("pages", {})
            for page_id in pages:
                if "thumbnail" in pages[page_id]:
                    return pages[page_id]["thumbnail"]["source"]
        except Exception:
            continue
            
    # 위키백과에 사진이 없는 경우 기본 프로필 실루엣 반환
    return "https://flaticon.com"

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding)

def cosine_similarity(a, b):
    dot_product = np.dot(b, a)
    norm_b = np.linalg.norm(b, axis=1)
    norm_a = np.linalg.norm(a)
    return dot_product / (norm_b * norm_a)

def analyze_with_gpt(user_summary, top3_df):
    leaders_info = ""
    for rank, (_, row) in enumerate(top3_df.iterrows(), 1):
        leaders_info += f"[{rank}위] {row['name']} ({row['era_region']})\n- 특징: {row['personality']}\n\n"

    prompt = f"""
당신은 심리 및 리더십 분석 전문가입니다.
사용자의 서술형 답변을 바탕으로, 가장 닮은 역사적 인물 1위, 2위, 3위가 선정되었습니다.

[사용자의 답변 요약]
{user_summary}

[선정된 1~3위 인물]
{leaders_info}

위 내용을 바탕으로 사용자에게 다음 형식에 맞춰 흥미롭고 깊이 있는 분석 결과를 작성해 주세요.
1. 전체적인 성향 총평 (2~3문장)
2. 1위 인물과의 공통점 및 닮은 이유 (사용자의 답변 내용을 구체적으로 인용하며 설명)
3. 2위, 3위 인물이 함께 나온 이유와 보완점
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 통찰력 있고 따뜻한 어조로 성향을 분석하는 심리·역사 전문가입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content

# --- 웹 화면 UI 구성 ---
st.title("🌍 나와 가장 닮은 세계의 역사적 지도자 찾기")
st.write("19세기~20세기 세계 각국의 지도자·사상가·혁신가 100명의 데이터베이스와 당신의 내면을 비교해 드립니다.")
st.divider()

df, leader_embeddings = load_data()

with st.form("quiz_form"):
    answers = []
    for q in QUESTIONS:
        ans = st.text_area(q, height=90, placeholder="평소 본인의 생각이나 경험을 편하게 적어주세요...")
        answers.append((q, ans.strip()))
    submitted = st.form_submit_button("🔍 나와 닮은 지도자 분석하기", use_container_width=True)

if submitted:
    if any(not ans for _, ans in answers):
        st.warning("5가지 질문에 모두 답변을 입력해 주세요!")
    else:
        with st.spinner("100명의 지도자와 당신의 행동 양식을 정밀 분석 중입니다..."):
            combined_answer = "\n\n".join([f"{q}\n답변: {ans}" for q, ans in answers])
            user_vec = get_embedding(combined_answer)

            scores = cosine_similarity(user_vec, leader_embeddings)
            df["similarity"] = scores
            top3 = df.sort_values(by="similarity", ascending=False).head(3)

            st.success("분석이 완료되었습니다!")
            st.subheader("🏆 당신과 가장 닮은 지도자 TOP 3")

            cols = st.columns(3)
            medals = ["🥇 1위", "🥈 2위", "🥉 3위"]
            for idx, (_, row) in enumerate(top3.iterrows()):
                with cols[idx]:
                    match_score = round(row["similarity"] * 100, 1)
                    st.metric(label=medals[idx], value=row["name"], delta=f"유사도 {match_score}")
                    
                    # 인물 대표 사진 출력 (토큰 비용 0원)
                    img_url = get_leader_image(row["name"])
                    st.image(img_url, use_container_width=True)
                    
                    st.caption(row["era_region"])

            st.divider()
            st.subheader("📋 심층 성향 분석 리포트")
            report = analyze_with_gpt(combined_answer, top3)
            st.markdown(report)
