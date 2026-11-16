import os
import itertools
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

# 검증 완료된 균형 잡힌 5지선다 + 6번 직접 기입 질문 세트
QUIZ_DATA = [
    {
        "question": "1. [위기 대처] 예상치 못한 큰 위기나 실패가 닥쳤을 때, 가장 먼저 어떻게 반응하시나요?",
        "options": [
            "① 오히려 투지가 불타오르며, 위험을 감수하더라도 정면으로 부딪혀 판도를 뒤집으려 한다.",
            "② 감정을 완전히 배제하고 현재 상황을 냉철한 원인과 데이터로 분석하며 빈틈을 찾는다.",
            "③ 성급한 행동을 자제하고, 고독한 침묵 속에서 상황이 가라앉을 때까지 인내하며 때를 기다린다.",
            "④ 고통받는 곁의 사람들을 먼저 품어 안고 다독이며, 함께 연대해 이겨낼 수 있도록 힘을 모은다.",
            "⑤ 기존의 방식이 통하지 않음을 인정하고, 전혀 새로운 발상이나 예술적·창의적 우회로를 실험한다.",
            "⑥ 기타 (직접 기입)"
        ]
    },
    {
        "question": "2. [의사결정] 중요한 결정을 내려야 할 때, 무엇을 가장 최우선 기준으로 삼으시나요?",
        "options": [
            "① 당장의 손해나 세상의 비난이 있더라도, 내가 옳다고 믿는 도덕적 양심과 지조를 끝까지 지킨다.",
            "② 추상적인 명분보다 현실적으로 가장 실속 있고 확실한 생존과 성과를 낼 수 있는 길을 택한다.",
            "③ 남들이 아직 보지 못하는 시대의 흐름과 내면의 강렬한 직감을 믿고 과감하게 승부를 건다.",
            "④ 결정으로 인해 소외되거나 상처받는 약자가 없도록, 서로 다른 목소리를 경청하며 화해를 이끈다.",
            "⑤ 작은 허점도 놓치지 않도록 모든 근거를 철저히 검증하고 실험해 본 뒤에야 비로소 움직인다.",
            "⑥ 기타 (직접 기입)"
        ]
    },
    {
        "question": "3. [갈등 해결] 주변 사람이나 조직과 의견이 크게 충돌할 때, 주로 어떤 태도를 취하시나요?",
        "options": [
            "① 애매하게 타협하기보다 무엇이 옳고 그른지 직설적이고 날카로운 언어로 명확히 짚고 넘어간다.",
            "② 감정적 대립을 피하고 양쪽의 입장을 끝까지 경청하며 모두가 수용할 수 있는 타협점을 조율한다.",
            "③ 물리적 충돌이나 거친 언쟁 대신, 비폭력적인 태도와 묵묵한 실천으로 상대의 양심을 움직인다.",
            "④ 집단의 광기나 다수의 의견에 억지로 맞추기보다, 고독을 감수하고 내 소신대로 내 길을 간다.",
            "⑤ 재치 있는 유머와 따뜻한 인간미로 날 선 긴장감을 먼저 녹인 뒤, 부드럽게 마음을 설득한다.",
            "⑥ 기타 (직접 기입)"
        ]
    },
    {
        "question": "4. [핵심 가치] 당신을 가장 뜨겁게 움직이게 만드는 삶의 원동력은 무엇인가요?",
        "options": [
            "① 기존의 낡은 관습과 억압을 깨부수고, 세상에 없던 새로운 기준과 자유를 창조하는 것.",
            "② 불평등과 고통에 맞서 소외된 약자를 돕고, 공동체의 정의와 평화를 위해 헌신하는 것.",
            "③ 세속적 성공이나 권력보다, 자연과 인간 내면의 근본 원리와 진리를 끝까지 파고들어 탐구하는 것.",
            "④ 내가 세운 목표를 최고 수준의 장인 정신으로 끌어올려, 타협 없는 완벽한 작품을 완성하는 것.",
            "⑤ 화려한 명예보다 소박한 일상 속에서 부끄러움 없는 양심을 지키며 내면의 평화를 가꾸는 것.",
            "⑥ 기타 (직접 기입)"
        ]
    },
    {
        "question": "5. [실행 방식] 새로운 일이나 프로젝트를 추진할 때, 평소 어떤 스타일로 일하시나요?",
        "options": [
            "① 책상 앞의 긴 고민보다 일단 현장에 뛰어들어 몸으로 부딪히고 수정하면서 길을 만들어간다.",
            "② 혼선이 생기지 않도록 전체 체계와 규칙, 일정을 빈틈없이 설계한 뒤 차근차근 실행한다.",
            "③ 권위적인 지시를 버리고 뛰어난 사람들과 수평적으로 토론하며, 신뢰를 바탕으로 권한을 나눈다.",
            "④ 남들이 지나치는 아주 작은 디테일 하나까지도 내 손으로 직접 확인하고 다듬어야 직성이 풀린다.",
            "⑤ 화려한 도약보다 매일 묵묵히 기본기를 다지고, 한 걸음씩 성실하게 쌓아 올리며 후대를 준비한다.",
            "⑥ 기타 (직접 기입)"
        ]
    }
]

@st.cache_data
def load_data_and_baseline():
    """100명 데이터와 Z-score 보정용 기준 분포(평균·표준편차)를 한 번만 계산해 캐싱합니다."""
    df = pd.read_csv(CSV_PATH)
    embeddings = np.load(EMB_PATH)
    leader_norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # 25개 객관식 보기로 기준 분포 계산 (쏠림 현상 방지용)
    base_options = [opt[2:].strip() for item in QUIZ_DATA for opt in item["options"][:5]]
    res = client.embeddings.create(model="text-embedding-3-small", input=base_options)
    opt_vecs = np.array([d.embedding for d in res.data]).reshape(5, 5, -1)

    all_raw = []
    for combo in itertools.product(range(5), repeat=5):
        u_vec = np.mean([opt_vecs[q][o] for q, o in enumerate(combo)], axis=0)
        u_norm = u_vec / np.linalg.norm(u_vec)
        all_raw.append(np.dot(leader_norms, u_norm))
    all_raw = np.array(all_raw)

    means = np.mean(all_raw, axis=0)
    stds = np.std(all_raw, axis=0) + 1e-8
    return df, leader_norms, means, stds

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
            title = search_results[0]["title"]

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
    return np.array(response.data[0].embedding)

def analyze_with_gpt(user_summary, top3_df):
    leaders_info = ""
    for rank, (_, row) in enumerate(top3_df.iterrows(), 1):
        leaders_info += f"[{rank}위] {row['name']} ({row['era_region']})\n- 특징: {row['personality']}\n\n"

    prompt = f"""
당신은 심리 및 리더십 분석 전문가입니다.
사용자의 답변을 바탕으로, 100명의 역사적 인물 중 가장 닮은 1위, 2위, 3위가 선정되었습니다.

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
    return response.choices[0].message.content

# --- 웹 화면 UI 구성 ---
st.title("🌍 나와 가장 닮은 세계의 역사적 지도자 찾기")
st.write("19세기~20세기 세계 각국의 지도자·사상가·혁신가 100명의 데이터베이스와 당신의 행동 양식을 비교해 드립니다.")
st.caption("가장 가까운 보기를 선택하거나, '⑥ 기타 (직접 기입)'을 눌러 나만의 생각을 자유롭게 적어주세요.")
st.divider()

df, leader_norms, means, stds = load_data_and_baseline()

collected_answers = []
missing_custom_input = False

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
        final_ans = selected[2:].strip()
        
    collected_answers.append((item["question"], final_ans))
    st.markdown("---")

if st.button("🔍 나와 닮은 지도자 분석하기", type="primary", use_container_width=True):
    if missing_custom_input:
        st.warning("'⑥ 기타 (직접 기입)'을 선택하신 항목의 내용을 입력해 주세요!")
    else:
        with st.spinner("100명의 지도자와 당신의 행동 양식을 정밀 분석 중입니다..."):
            # 순수 답변 내용만 결합하여 임베딩 (질문 제목 편향 제거)
            pure_answers_text = "\n".join([ans for _, ans in collected_answers])
            user_vec = get_embedding(pure_answers_text)
            user_norm = user_vec / np.linalg.norm(user_vec)

            # Z-score 정규화 점수로 100명 중 진정한 매칭 순위 계산
            raw_scores = np.dot(leader_norms, user_norm)
            z_scores = (raw_scores - means) / stds
            df["z_score"] = z_scores
            top3 = df.sort_values(by="z_score", ascending=False).head(3)

            st.success("분석이 완료되었습니다!")
            st.subheader("🏆 당신과 가장 닮은 지도자 TOP 3")

            cols = st.columns(3)
            medals = ["🥇 1위", "🥈 2위", "🥉 3위"]
            for idx, (_, row) in enumerate(top3.iterrows()):
                with cols[idx]:
                    # Z-score를 직관적인 매칭 지수(75~98% 범위)로 변환
                    match_pct = min(98.5, max(75.0, round(85.0 + row["z_score"] * 4.5, 1)))
                    st.metric(label=medals[idx], value=row["name"], delta=f"공명도 {match_pct}%")
                    
                    img_url = get_leader_image(row["name"])
                    st.image(img_url, use_container_width=True)
                    
                    st.caption(row["era_region"])

            st.divider()
            st.subheader("📋 심층 성향 분석 리포트")
            gpt_summary = "\n\n".join([f"{q}\n답변: {ans}" for q, ans in collected_answers])
            report = analyze_with_gpt(gpt_summary, top3)
            st.markdown(report)
