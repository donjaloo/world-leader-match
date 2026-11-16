import os
import itertools
import requests
import numpy as np
import pandas as pd
import urllib.parse
import streamlit as st
from openai import OpenAI

# 페이지 기본 설정
st.set_page_config(page_title="나와 닮은 역사 속 인물 찾기", page_icon="🎭", layout="centered")

# API 키 불러오기 (Streamlit Secrets 또는 로컬 환경변수)
api_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API 키가 설정되지 않았습니다. Secrets 설정을 확인해 주세요.")
    st.stop()

client = OpenAI(api_key=api_key)

# 모드별 질문 및 선택지 데이터
HERO_QUIZ = [
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

VILLAIN_QUIZ = [
    {
        "question": "1. [통제와 권력] 당신이 거대한 조직이나 집단의 절대적인 권한을 쥐게 된다면, 어떻게 다스릴까요?",
        "options": [
            "① 철저한 규칙과 감시 시스템을 구축해, 모든 구성원이 정해진 궤도에서 벗어나지 못하게 통제한다.",
            "② 화려한 연설과 카리스마로 대중의 감정과 자부심을 자극해, 스스로 나를 숭배하고 따르게 만든다.",
            "③ 측근들이 서로 경쟁하고 견제하도록 판을 짜서, 오직 나에게만 충성하도록 권력 균형을 조종한다.",
            "④ 직접 악역을 맡기보다 대리인을 내세워 궂은일을 처리하게 하고, 나는 온화하고 정의로운 얼굴을 유지한다.",
            "⑤ 기존의 낡은 제도와 기득권을 단숨에 뒤엎고, 내 머릿속의 완벽한 이상 사회를 강압적으로 실현한다.",
            "⑥ 기타 (직접 기입)"
        ]
    },
    {
        "question": "2. [배신과 위협] 믿었던 측근이나 동료가 당신의 권위에 도전하거나 배신하려 할 때, 어떻게 대처하나요?",
        "options": [
            "① 감정을 드러내지 않고 차갑게 때를 기다렸다가, 가장 치명적인 순간에 흔적도 없이 숙청한다.",
            "② 본보기가 되도록 공개적이고 가혹한 처벌을 내려, 다른 누구도 감히 반항할 엄두를 못 내게 만든다.",
            "③ 상대의 약점과 비밀을 미리 수집해 두었다가, 조용히 고립시키고 스스로 무너지도록 압박한다.",
            "④ 즉각적인 분노와 무력으로 제압해, 나를 거스르는 대가가 얼마나 참혹한지 똑똑히 각인시킨다.",
            "⑤ 거짓 미소와 유화책으로 상대를 방심하게 만든 뒤, 결정적인 순간에 뒤통수를 쳐서 판을 뒤집는다.",
            "⑥ 기타 (직접 기입)"
        ]
    },
    {
        "question": "3. [목적과 수단] 원하는 목표를 달성하기 위해 도덕적 선을 넘어야 할 때, 어떤 태도를 취하나요?",
        "options": [
            "① 큰 대의와 효율을 위해서라면 소수의 희생이나 비정한 수단도 당연히 감수해야 할 비용이다.",
            "② 내가 믿는 절대적 원칙과 순수성을 지키는 일이라면, 어떤 타협이나 관용도 허용하지 않는다.",
            "③ 법과 절차는 약자를 통제하기 위한 도구일 뿐, 나는 그 규칙 위에 군림하며 실속과 이익을 챙긴다.",
            "④ 대중에게는 화려한 명분과 구원을 약속하면서, 뒤에서는 내 사적인 욕망과 부를 철저히 채운다.",
            "⑤ 상부의 지시나 조직의 시스템이라는 명분 뒤에 숨어, 죄책감 없이 차갑고 성실하게 임무를 집행한다.",
            "⑥ 기타 (직접 기입)"
        ]
    },
    {
        "question": "4. [내면의 욕망] 당신의 내면 깊숙한 곳에 숨겨진 가장 강렬한 어두운 욕망은 무엇인가요?",
        "options": [
            "① 누구도 나를 무시하거나 위협하지 못하도록, 모든 변수와 사람을 내 손아귀에 완벽히 장악하는 것.",
            "② 역사에 내 이름을 영원히 각인시키고, 거대한 기념비와 화려한 영광으로 만인의 우러러봄을 받는 것.",
            "③ 나를 얕보거나 상처 입혔던 세상의 모든 위선과 기존 질서를 철저히 파괴하고 복수하는 것.",
            "④ 세속의 도덕이나 타인의 시선에 얽매이지 않고, 오직 내 취향과 쾌락, 변덕대로 세상을 휘두르는 것.",
            "⑤ 아무도 믿지 못하는 고독한 성벽 안에서, 외부의 모든 위협을 차단한 채 나만의 왕국을 지키는 것.",
            "⑥ 기타 (직접 기입)"
        ]
    },
    {
        "question": "5. [위기와 몰락] 사방이 적으로 둘러싸이고 패배가 눈앞에 닥쳤을 때, 마지막으로 어떤 선택을 할까요?",
        "options": [
            "① 현실적인 후퇴나 항복은 없다. 마지막 한 사람까지 동원해 파멸적인 정면 돌파를 밀어붙인다.",
            "② 현실의 참혹한 패배를 외면한 채, 끝까지 내가 승리하고 있다는 환상과 선동 속에서 자존심을 지킨다.",
            "③ 냉정하게 손익을 계산해 재빨리 태세를 전환하고, 살아남기 위해 어제의 적과도 결탁한다.",
            "④ 내가 가질 수 없다면 차라리 모든 것을 불태워버리겠다는 극단적인 파괴와 보복을 선택한다.",
            "⑤ 비밀 통로와 숨겨둔 자금을 이용해 조용히 자취를 감춘 뒤, 배후에서 다시 재기의 칼을 간다.",
            "⑥ 기타 (직접 기입)"
        ]
    }
]

@st.cache_data
def load_mode_data(mode):
    """선택된 모드(영웅 또는 빌런)에 맞는 CSV, 벡터, 기준 분포를 불러옵니다."""
    if mode == "hero":
        csv_path, emb_path, quiz = "leaders_100.csv", "leader_embeddings.npy", HERO_QUIZ
    else:
        csv_path, emb_path, quiz = "villains_100.csv", "villain_embeddings.npy", VILLAIN_QUIZ

    df = pd.read_csv(csv_path)
    embeddings = np.load(emb_path)
    norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # 25개 객관식 보기로 기준 분포(Z-score용) 계산
    base_options = [opt[2:].strip() for item in quiz for opt in item["options"][:5]]
    res = client.embeddings.create(model="text-embedding-3-small", input=base_options)
    opt_vecs = np.array([d.embedding for d in res.data]).reshape(5, 5, -1)

    all_raw = []
    for combo in itertools.product(range(5), repeat=5):
        u_vec = np.mean([opt_vecs[q][o] for q, o in enumerate(combo)], axis=0)
        u_norm = u_vec / np.linalg.norm(u_vec)
        all_raw.append(np.dot(norms, u_norm))
    all_raw = np.array(all_raw)

    means = np.mean(all_raw, axis=0)
    stds = np.std(all_raw, axis=0) + 1e-8
    return df, norms, means, stds

@st.cache_data(show_spinner=False)
def get_person_info(name):
    """위키백과 API에서 인물 대표 이미지 URL과 백과사전 페이지 URL을 무료(토큰 0원)로 가져옵니다."""
    headers = {"User-Agent": "HistoryPersonaMatchApp/1.0 (contact@example.com)"}
    default_img = "https://flaticon.com"
    default_url = f"https://wikipedia.org{urllib.parse.quote(name)}"
    
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
            page_url = f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"

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
                    return pages[page_id]["thumbnail"]["source"], page_url
            
            # 사진은 없지만 문서는 찾은 경우 해당 문서 URL 반환
            return default_img, page_url
        except Exception:
            continue
            
    return default_img, default_url

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding)

def analyze_with_gpt(mode, user_summary, top3_df):
    leaders_info = ""
    for rank, (_, row) in enumerate(top3_df.iterrows(), 1):
        leaders_info += f"[{rank}위] {row['name']} ({row['era_region']})\n- 특징: {row['personality']}\n\n"

    if mode == "hero":
        role_desc = "당신은 통찰력 있고 따뜻한 어조로 리더십과 성향을 분석하는 심리·역사 전문가입니다."
        guide_text = """
1. 전체적인 리더십 및 성향 총평 (2~3문장)
2. 1위 인물과의 공통점 및 닮은 이유 (사용자의 선택·답변을 구체적으로 인용하며 강점 중심으로 설명)
3. 2위, 3위 인물이 함께 나온 이유와 앞으로 발휘하면 좋을 시너지
"""
    else:
        role_desc = "당신은 인간 내면의 권력욕과 그림자 심리를 예리하고 흥미롭게 해부하는 다크 심리 분석가입니다."
        guide_text = """
1. 내면에 숨겨진 흑화(다크) 성향 총평 (2~3문장, 흥미롭고 몰입감 있는 톤으로 작성)
2. 1위 빌런·권력자와 닮은 지점 (사용자의 선택·답변을 구체적으로 짚어주며 어떤 권력·통제 스타일을 지녔는지 분석)
3. 2위, 3위 인물이 함께 나온 이유 및 현실에서 경계해야 할 '흑화 주의보' 조언
"""

    prompt = f"""
사용자의 답변을 바탕으로, 100명의 역사적 인물 중 가장 닮은 1위, 2위, 3위가 선정되었습니다.

[사용자의 선택 및 답변 요약]
{user_summary}

[선정된 1~3위 인물]
{leaders_info}

위 내용을 바탕으로 사용자에게 다음 형식에 맞춰 깊이 있고 흥미로운 분석 결과를 작성해 주세요.
{guide_text}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": role_desc},
            {"role": "user", "content": prompt}
        ],
        temperature=0.75
    )
    return response.choices[0].message.content

# --- 웹 화면 UI 구성 ---
st.title("🎭 나와 닮은 역사 속 인물 찾기")
st.write("200명의 방대한 역사적 인물 데이터베이스를 통해 당신의 빛과 그림자를 분석합니다.")

# 모드 선택 스위치
selected_mode_label = st.radio(
    "테스트 모드를 선택하세요:",
    ["🌟 영웅·지도자 모드 (100인)", "🖤 다크·빌런 모드 (100인)"],
    horizontal=True
)
current_mode = "hero" if "영웅" in selected_mode_label else "villain"

if current_mode == "hero":
    st.info("🌟 **영웅·지도자 모드**: 위기 돌파, 신념, 포용, 혁신을 이끈 100명의 위인들과 당신의 행동 양식을 비교합니다.")
    active_quiz = HERO_QUIZ
else:
    st.warning("🖤 **다크·빌런 모드**: 역사를 뒤흔든 100명의 폭군·독재자·책략가들과 당신 내면의 권력 및 통제 심리를 비교합니다.")
    active_quiz = VILLAIN_QUIZ

st.caption("가장 가까운 보기를 선택하거나, '⑥ 기타 (직접 기입)'을 눌러 나만의 생각을 자유롭게 적어주세요.")
st.divider()

df, norms, means, stds = load_mode_data(current_mode)

collected_answers = []
missing_custom_input = False

for idx, item in enumerate(active_quiz):
    st.subheader(item["question"])
    selected = st.radio(
        label=f"{current_mode}_질문_{idx+1}",
        options=item["options"],
        key=f"{current_mode}_radio_{idx}",
        label_visibility="collapsed"
    )
    
    if selected.startswith("⑥"):
        custom_text = st.text_area(
            label="직접 기입",
            key=f"{current_mode}_custom_{idx}",
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

btn_label = "🔍 나와 닮은 영웅·지도자 분석하기" if current_mode == "hero" else "🖤 내 안의 다크·빌런 성향 분석하기"

if st.button(btn_label, type="primary", use_container_width=True):
    if missing_custom_input:
        st.warning("'⑥ 기타 (직접 기입)'을 선택하신 항목의 내용을 입력해 주세요!")
    else:
        with st.spinner("100명의 데이터베이스와 당신의 심리를 정밀 대조 중입니다..."):
            pure_answers_text = "\n".join([ans for _, ans in collected_answers])
            user_vec = get_embedding(pure_answers_text)
            user_norm = user_vec / np.linalg.norm(user_vec)

            raw_scores = np.dot(norms, user_norm)
            z_scores = (raw_scores - means) / stds
            df["z_score"] = z_scores
            top3 = df.sort_values(by="z_score", ascending=False).head(3)

            st.success("분석이 완료되었습니다!")
            result_title = "🏆 당신과 가장 닮은 지도자 TOP 3" if current_mode == "hero" else "🖤 당신 내면과 닮은 다크 리더 TOP 3"
            st.subheader(result_title)

            cols = st.columns(3)
            medals = ["🥇 1위", "🥈 2위", "🥉 3위"]
            for idx, (_, row) in enumerate(top3.iterrows()):
                with cols[idx]:
                    match_pct = min(98.5, max(75.0, round(85.0 + row["z_score"] * 4.5, 1)))
                    st.metric(label=medals[idx], value=row["name"], delta=f"공명도 {match_pct}%")
                    
                    img_url, wiki_url = get_person_info(row["name"])
                    st.image(img_url, use_container_width=True)
                    
                    # 클릭 시 새 탭에서 위키백과 인물 설명 열기
                    st.markdown(f"📖 **[{row['name']} 알아보기 ↗]({wiki_url})**")
                    st.caption(row["era_region"])

            st.divider()
            st.subheader("📋 심층 성향 분석 리포트")
            gpt_summary = "\n\n".join([f"{q}\n답변: {ans}" for q, ans in collected_answers])
            report = analyze_with_gpt(current_mode, gpt_summary, top3)
            st.markdown(report)

