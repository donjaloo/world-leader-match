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

# 5지선다 + 6번 직접 기입 질문 세트
QUIZ_DATA = [
    {
        "question": "1. [위기 대처] 예상치 못한 큰 위기나 실패가 닥쳤을 때, 가장 먼저 어떻게 반응하시나요?",
        "options": [
            "① 오히려 투지가 불타오르며, 위험을 감수하더라도 정면으로 부딪혀 판도를 뒤집으려 한다.",
            "② 감정을 완전히 배제하고 현재 상황을 객관적인 데이터와 원인으로 냉정하게 분석한다.",
            "③ 성급한 행동을 자제하고, 상황이 가라앉을 때까지 인내심을 갖고 다각도로 숙고하며 때를 기다린다.",
            "④ 주변 사람들의 불안감을 먼저 다독이고, 함께 힘을 모아 이겨낼 수 있도록 분위기를 이끈다.",
            "⑤ 기존의 방식이 통하지 않음을 인정하고, 전혀 새로운 발상이나 우회로를 즉각 실험해 본다.",
            "⑥ 기타 (직접 기입)"
        ]
    },
    {
        "question": "2. [의사결정] 중요한 결정을 내려야 할 때, 무엇을 가장 최우선 기준으로 삼으시나요?",
        "options": [
            "① 당장의 손해나 주변의 비난이 있더라도, 내가 옳다고 믿는 신념과 양심을 끝까지 지킨다.",
            "② 추상적인 명분보다 현실적으로 가장 실속 있고 확실한 성과를 낼 수 있는 길을 택한다.",
            "③ 남들이 아직 보지 못하는 시대의 흐름과 내면의 강렬한 직감을 믿고 과감하게 승부를 건다.",
            "④ 결정으로 인해 소외되거나 상처받는 사람이 없도록, 주변의 목소리를 경청하며 조화를 이룬다.",
            "⑤ 작은 허점이나 위험 요소도 놓치지 않도록 모든 근거를 꼼꼼히 따져본 뒤에야 비로소 움직인다.",
            "⑥ 기타 (직접 기입)"
    },
    {
        "question": "3. [갈등 해결] 주변 사람이나 조직과 의견이 크게 충돌할 때, 주로 어떤 태도를 취하시나요?",
        "options": [
            "① 애매하게 타협하기보다 무엇이 옳고 그른지 직설적이고 논리적으로 명확히 짚고 넘어간다.",
            "② 감정적 대립을 피하고 양쪽의 입장을 끝까지 경청하며 모두가 수용할 수 있는 타협점을 조율한다.",
            "③ 물리적 충돌이나 거친 언쟁 대신, 흔들리지 않는 태도와 묵묵한 행동으로 상대의 양심을 움직인다.",
            "④ 집단의 분위기나 다수의 의견에 억지로 맞추기보다, 내 소신대로 고독하게 내 길을 간다.",
            "⑤ 유머와 여유로운 태도로 날 선 긴장감을 먼저 누그러뜨린 뒤, 부드럽게 내 편으로 설득한다.",
            "⑥ 기타 (직접 기입)"
        ]
    },
    {
        "question": "4. [핵심 가치] 당신을 가장 뜨겁게 움직이게 만드는 삶의 원동력은 무엇인가요?",
        "options": [
            "① 기존의 낡은 틀과 관습을 깨부수고, 세상에 없던 새로운 기준과 시스템을 창조하는 것.",
            "② 불의와 불평등에 맞서 약자를 보호하고, 더 나은 공동체와 대의명분을 위해 헌신하는 것.",
            "③ 세속적인 성공이나 권력보다, 세상의 근본 원리와 진리를 끝까지 파고들어 이해하는 것.",
            "④ 내가 세운 목표를 최고 수준의 완성도로 끌어올려, 완벽한 결과물과 시스템을 구축하는 것.",
            "⑤ 누구의 간섭이나 통제도 받지 않고, 온전히 나다운 삶을 주체적으로 설계하고 표현하는 것.",
            "⑥ 기타 (직접 기입)"
        ]
    },
    {
        "question": "5. [실행 방식] 새로운 일이나 프로젝트를 추진할 때, 평소 어떤 스타일로 일하시나요?",
        "options": [
            "① 책상 앞의 긴 고민보다 일단 현장에 뛰어들어 직접 부딪히고 수정하면서 길을 만들어간다.",
            "② 혼선이 생기지 않도록 전체 흐름과 역할, 일정을 짜임새 있게 설계한 뒤 차근차근 실행한다.",
            "③ 권위적인 지시를 버리고 동료들과 자유롭게 토론하며, 서로를 신뢰하고 권한을 나눈다.",
            "④ 남들이 지나치는 아주 작은 디테일 하나까지도 내 손으로 직접 확인하고 다듬어야 직성이 풀린다.",
            "⑤ 화려한 도약보다 매일 묵묵히 기본기를 다지고, 꾸준한 성실함으로 멀리 내다보며 쌓아간다.",
            "⑥ 기타 (직접 기입)"
        ]
    }
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
            
            title = search_results[0]["title"]  # [0] 추가!
            
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
            
    return "https://flaticon.com"

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding)  # [0] 추가!

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
사용자의 답변을 바탕으로, 가장 닮은 역사적 인물 1위, 2위, 3위가 선정되었습니다.

[사용자의 선택 및 답변 요약]
{user_summary}

[선정된 1~3위 인물]
{leaders_info}

위 내용을 바탕으로 사용자에게 다음 형식에 맞춰 흥미롭고 깊이 있는 분석 결과를 작성해 주세요.
1. 전체적인 성향 총평 (2~3문장)
2. 1위 인물과의 공통점 및 닮은 이유 (사용자가 선택한 문구 또는 직접 작성한 내용을 구체적으로 언급하며 설명)
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
    return response.choices[0].message.content  # [0] 추가!

# --- 웹 화면 UI 구성 ---
st.title("🌍 나와 가장 닮은 세계의 역사적 지도자 찾기")
st.write("19세기~20세기 세계 각국의 지도자·사상가·혁신가 100명의 데이터베이스와 당신의 행동 양식을 비교해 드립니다.")
st.caption("가장 가까운 보기를 선택하거나, '⑥ 기타 (직접 기입)'을 눌러 나만의 생각을 자유롭게 적어주세요.")
st.divider()

df, leader_embeddings = load_data()

collected_answers = []
missing_custom_input = False

# 실시간 입력창 활성화를 위해 st.form 대신 일반 위젯 배치
for idx, item in enumerate(QUIZ_DATA):
    st.subheader(item["question"])
    selected = st.radio(
        label=f"질문 {idx+1} 선택",
        options=item["options"],
        key=f"radio_{idx}",
        label_visibility="collapsed"
    )
    
    if selected.startswith("⑥"):
        custom_text = st.text_area(
            label="직접 기입",
            key=f"custom_{idx}",
            height=80,
            placeholder="위 보기에 없는 나만의 생각이나 행동 방식을 자유롭게 적어주세요...",
            label_visibility="collapsed"
        )
        final_ans = custom_text.strip()
        if not final_ans:
            missing_custom_input = True
    else:
        # 앞의 번호 기호(①, ② 등)를 제거하고 순수 문장만 추출
        final_ans = selected[2:].strip()
        
    collected_answers.append((item["question"], final_ans))
    st.markdown("---")

if st.button("🔍 나와 닮은 지도자 분석하기", type="primary", use_container_width=True):
    if missing_custom_input:
        st.warning("'⑥ 기타 (직접 기입)'을 선택하신 항목의 내용을 입력해 주세요!")
    else:
        with st.spinner("100명의 지도자와 당신의 행동 양식을 정밀 분석 중입니다..."):
            combined_answer = "\n\n".join([f"{q}\n답변: {ans}" for q, ans in collected_answers])
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
                    
                    img_url = get_leader_image(row["name"])
                    st.image(img_url, use_container_width=True)
                    
                    st.caption(row["era_region"])

            st.divider()
            st.subheader("📋 심층 성향 분석 리포트")
            report = analyze_with_gpt(combined_answer, top3)
            st.markdown(report)
