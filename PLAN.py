import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
import re
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from docx import Document
import tempfile
import os
import ollama
from google.cloud import vision
import io
from PIL import Image
from difflib import SequenceMatcher 
from dotenv import load_dotenv
# ✅ 初始化 Gemini API
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY","")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# ✅ 載入 FAISS 向量庫
INDEX_FILE_PATH = "faiss_index"


from langchain_community.vectorstores import FAISS

# 修改後
import torch



vector_store = FAISS.load_local(INDEX_FILE_PATH, embeddings=HuggingFaceEmbeddings(), allow_dangerous_deserialization=True)


import os
import json
from google.cloud import vision
from google.oauth2 import service_account
from google.cloud import vision

import os
import json
import tempfile
import streamlit as st
from google.cloud import vision

import os









# ✅ 在主程式中呼叫
# ✅ 分類提示詞
FACILITY_PROMPT = """
請判斷下列申請文件所屬醫療機構層級，僅從下列五類中擇一，**不得恣意新增新分類，也不得將醫學會等組織視為醫學中心**：
「醫學中心」、「關鍵基礎設施區域醫院」、「非關鍵基礎設施區域醫院」、「地區醫院」、「診所」。
請嚴格只回覆其中一個分類名稱，不要加任何說明或標點符號。

若文件中提及的區域醫院名稱屬於以下清單，請直接分類為「關鍵基礎設施區域醫院」：
振興醫療財團法人振興醫院
臺北醫學大學附設醫院
醫療財團法人羅許基金會羅東博愛醫院
長庚醫療財團法人基隆長庚紀念醫院
佛教慈濟醫療財團法人台北慈濟醫院
衛生福利部雙和醫院
天主教耕莘醫療財團法人耕莘醫院
東元綜合醫院
大千綜合醫院
澄清綜合醫院中港分院
童綜合醫療社團法人童綜合醫院
光田醫療社團法人光田綜合醫院
秀傳醫療社團法人秀傳紀念醫院
秀傳醫療財團法人彰濱秀傳紀念醫院
長庚醫療財團法人嘉義長庚紀念醫院
佛教慈濟醫療財團法人大林慈濟醫院
戴德森醫療財團法人嘉義基督教醫院
義大醫療財團法人義大醫院
安泰醫療社團法人安泰醫院
臺北市立聯合醫院
衛生福利部桃園醫院
國立台灣大學附設醫院分院新竹醫院
國立台灣大學附設醫院雲林分院
連江縣立醫院
衛生福利部金門醫院
衛生福利部南投醫院
衛生福利部澎湖醫院
衛生福利部臺中醫院
衛生福利部臺南醫院
衛生福利部屏東醫院
衛生福利部花蓮醫院

若文件中提及的區域醫院名稱不屬於清單內，請直接分類為「非關鍵基礎設施區域醫院」：

---文件內容---
{text}
---結束---
"""

REPEATED_SUBSIDY_PROMPT = """
請在目錄頁或申請單位自我檢核項目表中檢查是否有附上「未有重複申請計畫之聲明切結書」，若沒有附上應視為有「重複申請補助」之情事

是否重複申請：1/0
理由：（簡要說明判斷依據）

---文件內容---
{text}
---結束---
"""

PLAN_PROMPT = """
請依據申請書原文，準確擷取並列出下列資訊，**請勿將範疇或子標題誤判為計畫名稱**：

申請機構：XXX  
計畫名稱：XXX（注意：請勿誤將計畫範疇誤認為計畫名稱）  
計畫申請編號：XXX（應為兩至三位數字）

請嚴格依照上述格式作答，**不得包含任何多餘字元或說明**，也不得有任何錯字或格式偏差。

---文件內容---
{text}
---結束---
"""

FACILITY_SCORE_CAP_INFO = {
    "醫學中心": 30,
    "關鍵基礎設施區域醫院": 30,
    "非關鍵基礎設施區域醫院": 26,
    "地區醫院": 23,
    "診所": 23
}

INFO_QUESTIONS = [
    "1.是否設置機構資通安全長(CISO)、資安專責人員？",
    "2.是否訂定、修正及實施資通安全維護計畫？",
    "3.是否辦理資通訊系統盤點及風險評估？",
    "4.是否已加入「衛生福利部資安與資訊分享與分析中心？",
    "5.是否落實服務供應商管理，包含將資安納入契約及委外前進行資安風險評估？",
    "6.是否要求所有人員每年參加至少3小時資通安全通識教育訓練？",
    "7.是否將網路區隔？",
    "8.是否將重要資料備份及加密？",
    "9.是否於使用者電腦及伺服器建置防毒軟體？",
    "10.是否建置網路防火牆？",
    "11.是否辦理弱點掃描？",
    "12.是否辦理滲透測試？",
    "13.是否已建置是否建置電子郵件過濾機制(SPAM)？",
    "14.是否已建置入侵偵測及防禦機制？",
    "15.是否已建置應用程式防火牆？",
    "16.是否已建置進階持續性威脅攻擊防禦措施？",
    "17.是否已建置資通安全威脅偵測管理機制(SOC)？",
    "18.是否已導入政府組態基準(GCB)？",
    "19.是否已導入端點偵測及應變機制(EDR)？",
    "20.是否曾於本部或資通安全署資通安全稽核表現績優？",
    "21.是否曾參加本部紅藍隊攻防演練(包含參加評選)？",
    "22.是否列管並逐年汰換大陸廠排資通訊產品？",
    "23.是否建立 FHIR 結構資料庫（符合 TW Core IG）？",
    "24.是否有跨院資料交換規劃？",
    "25.是否設有資料治理流程/SOP 或清理轉換機制？",
    "26.是否能與 SMART on FHIR、CDS hook、CQL 接軌？",
    "27.是否採用國際標準如 LOINC、SNOMED CT、RxNorm？",
    "28.是否在醫療數據資料應用中，包含TW CDI資料集？",
    "29.是否提及七大倫理原則（自主、透明、當責、安全、公平、永續、隱私）？",
    "30.是否設有 AI 管理委員會與制度規章？",
    "31.是否導入 AI 透明性與揭露機制？",
    "32.是否有 AI 落地治理管理辦法？",
    "33.是否進行模型效能監測與再訓練？",
    "34.是否有實際 AI 應用案例？",
    "35.是否依據負責任AI 中心規範進行對標建置？",
    "36.是否依據負責任AI 中心規範進行對標建置？"
]

def extract_text_by_line(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    lines = []
    for page in doc:
        blocks = page.get_text("blocks")
        for b in blocks:
            text = b[4].strip()
            if text:
                lines.append(text)
    return "\n\n".join(lines)

import fitz  # PyMuPDF
import fitz
from PIL import Image





        

def build_individual_prompts(questions, full_text, role_name="審查委員"):
    prompts = []

    for q in questions:
        docs = vector_store.similarity_search(q, k=3)
        rag_context = "\n---\n".join(doc.page_content for doc in docs)
        prompt = f"""
你是一位{role_name}，請閱讀下方的文件內容，針對指定題目參考智慧醫療中心技術手冊作答。  
請注意：若文件中未明確提及指定內容，請勿依據其他字眼合理推測、延伸、補充或給分
請針對該題回答「得分（只能是 0 或 1 分）」以及「原因」。

⚠️ 請務必嚴格遵守以下規則，否則系統將無法解析你的回答：
1. 不要使用任何 Markdown 語法（例如：**粗體**、`程式碼區塊`、```區塊符號```、- 項目符號等）
2. 請勿添加任何額外說明、問候語、總結語或格式變化
3. 回答必須完全依照下列格式，否則會被視為格式錯誤

審查準則（🚫 嚴禁推論與誤判）：
1. 不得基於推論給分（例如：不能因為文件提到 AI 或資安字詞就合理化未提及的內容）
2. 不得以詞語相似或模糊描述作為符合依據（例如：「環境永續」≠「七大倫理中的永續」）
3. 僅能根據文件中**明確陳述**的事證判斷，不可使用 AI 合理化或幻覺內容
4. 認證標準如「ISO 27001」可視為**已執行風險評估**，但不得推論已建置 WAF、防火牆等
5. 如果僅提到平台建置、LINE 帳號、合作等，不代表有「跨院資料交換規劃」
6. 文件中提及使用 EDR ≠ 已建置防火牆、SPAM 過濾、或弱點掃描等基礎機制
7. 「資通安全維護計畫」應為專門文件，不能以「內文提到資安規劃」作為替代依據

回答格式如下：
得分 ⟪x⟫/1 ，原因：...(請緊接在同一行)

題目：{q}
---智慧醫療中心技術手冊---
{rag_context}
---文件內容開始---
{full_text}
---文件內容結束---
"""
        prompts.append(prompt.strip())
    return prompts

def get_gemini_response(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ 發生錯誤：{e}"


from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document

def query_rag_for_legal_compliance(full_text, vector_store=vector_store, model=model):
    query = "本文件是否符合政府補助計畫的適法性？請根據法規與文件內容給出明確結論與簡要理由。"
    docs = vector_store.similarity_search(query, k=3)
    rag_context = "\n---\n".join(doc.page_content for doc in docs)
    rag_prompt = f"""
你是一位熟悉政府補助法規與實務審查的資深審查員，請依據下列法規與技術規範，嚴格審查下方申請書內容是否符合政府補助之「資安要求」與「法規合規性」。
⚠️ 請勿基於推論和非明載內容給予符合評價，並請勿將未提及之法規作為否決依據。
⚠️ 請嚴格依據以下文件進行審查，不要使用任何 Markdown 語法（例如：**粗體**、`程式碼區塊`、```區塊符號```、- 項目符號等），也不得推論、臆測、合理化或補足未明載於計畫中的內容：
- 衛福部《健康台灣深耕計畫》範疇三：資安治理、資料治理、AI治理 懶人包與問答集
- 《資通安全管理法》與《資通安全責任等級分級辦法》
- 《個人資料保護法》、《醫療機構電子病歷製作與管理辦法》
- 《衛福部醫療資通系統資安防護基準》

請依據下列三大構面與醫療機構層級，逐項判斷其是否符合適法性標準：

一、基本適法性標準（所有醫療機構皆須符合）
二、等級責任標準（資安法納管 vs 非納管醫療機構）
三、特定系統與資料使用適法性（如 AI 導入、資料交換、跨院共享）

---向量查詢結果（法規文件段落）---
{rag_context}

---完整申請書節錄---
{full_text}

請依以下格式委婉作答，務必保持格式正確：

是否符合適法性：符合／不符合  
理由：（請具體列出不符合之條文或缺漏事項，並明確說明其違反的法規或政策標準，並對應其醫療機構層級要求）
"""
    response = model.generate_content(rag_prompt)
    return response.text.strip()

def extract_table_from_response(response_text, questions, facility_level="", score_cap_dict=None, repeated_subsidy_result=None, plan_result=None, legal_compliance_result=None):
    rows = []

    scores = re.findall(
        r"得分\s*⟪(\d{1,2})⟫\s*/1\s*[，,:：]\s*原因[:：]?\s*(.+?)(?=\n*得分\s*⟪|\Z)",
        response_text,
        flags=re.DOTALL
    )

    if len(scores) < len(questions):
        scores += [("0", "⚠️ Gemini 未回覆此題")] * (len(questions) - len(scores))
    elif len(scores) > len(questions):
        scores = scores[:len(questions)]

    for i, (score, reason) in enumerate(scores):
        cleaned_reason = reason.strip()
        rows.append({"題目": questions[i], "得分": int(score), "原因": cleaned_reason})

    # ✅ 篩除虛擬題目
    rows = [row for row in rows if not row["題目"].startswith("36.")]

    # 分三領域計算加權分數
    df = pd.DataFrame(rows)
    df["領域"] = ""

    # 分配領域標籤
    df.loc[0:21, "領域"] = "資安治理"
    df.loc[22:27, "領域"] = "資料治理"
    df.loc[28:34, "領域"] = "AI治理"

    # 各領域分數加總與加權
    score_dict = {
        "資安治理": {"total": 22, "weight": 40},
        "資料治理": {"total": 6, "weight": 30},
        "AI治理": {"total": 7, "weight": 30},
    }

    summary_rows = []
    weighted_total = 0
    for domain, config in score_dict.items():
        domain_score = df[df["領域"] == domain]["得分"].sum()
        weighted = round(domain_score / config["total"] * config["weight"])
        weighted_total += weighted
        summary_rows.append({
            "題目": f"📊 {domain}總分",
            "得分": f"{domain_score}/{config['total']} → {weighted}分",
            "原因": ""
        })

    summary_rows.append({
        "題目": "✅ 加權總分",
        "得分": f"{weighted_total}/100",
        "原因": f"{facility_level} 計算完成"
    })


    # ✅ 重複補助
    if repeated_subsidy_result:
        sub_score = 0 if "是否重複申請：0" in repeated_subsidy_result else ""
        summary_rows.append({
            "題目": "📌 是否重複申請補助",
            "得分": sub_score,
            "原因": repeated_subsidy_result
        })

    # ✅ 機構、計畫名稱、計畫代號
    if plan_result:
        org, plan_name, plan_code = parse_plan_info(plan_result)
        summary_rows.append({"題目": "🏢 申請機構", "得分": "", "原因": org})
        summary_rows.append({"題目": "📝 計畫名稱", "得分": "", "原因": plan_name})
        summary_rows.append({"題目": "🆔 計畫申請編號", "得分": "", "原因": plan_code})

    # ✅ 適法性結果
    if legal_compliance_result:
        legal_score = "符合" if "是否符合適法性：符合" in legal_compliance_result else "不符合"
        summary_rows.append({
            "題目": "📌 是否符合適法性",
            "得分": legal_score,
            "原因": legal_compliance_result
        })

    return pd.DataFrame(summary_rows)


def render_score_table(title, response_text, questions, color, facility_level, score_cap_dict, pdf_filename, repeated_subsidy_result, plan_result,legal_compliance_result):
    df = extract_table_from_response(response_text, questions, facility_level, score_cap_dict, repeated_subsidy_result, plan_result,legal_compliance_result)
    st.markdown(f"<h4 style='color:{color}'>{title}</h4>", unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)
    csv_data = df.to_csv(index=False)
    filename = f"{pdf_filename}_資訊處.csv"
    with st.container():
        st.download_button(
            label=f"📥 下載 {filename}",
            data=csv_data,
            file_name=filename,
            mime="text/csv",
            key=f"download_{pdf_filename}"  # 🔑 使用唯一 key 防止重新觸發 rerun
        )
    st.markdown(f"🔍 Gemini 回傳字數：{len(response_text)}")
    with st.expander("📝 查看原始 Gemini 回應"):
        st.write(response_text)

def parse_plan_info(plan_result_text):
    org, name, code = "❓ 無法判斷", "❓ 無法判斷", "❓ 無法判斷"

    # 使用 MULTILINE 模式，確保只抓開頭的行
    org_match = re.search(r"^申請機構[：:]\s*(.+)$", plan_result_text, flags=re.MULTILINE)
    name_match = re.search(r"^計畫名稱[：:]\s*(.+)$", plan_result_text, flags=re.MULTILINE)
    code_match = re.search(r"^(計畫申請編號|計畫編號)[：:]\s*(.+)$", plan_result_text, flags=re.MULTILINE)

    if org_match:
        org = org_match.group(1).strip()
    if name_match:
        name = name_match.group(1).strip()
    if code_match:
        code = code_match.group(2).strip()

    return org, name, code

def main():
    st.set_page_config("📄 多 PDF 自動分析器", layout="wide")
    st.title("📄 多份 PDF 自動分析與資訊處評分")
    uploaded_pdfs = st.file_uploader("📥 上傳 PDF 文件（可複選）", type=["pdf"], accept_multiple_files=True)
    use_split = st.checkbox("✅ 啟用逐題分析（建議用於長文件）")
    if uploaded_pdfs and st.button("🚀 開始分析"):
        for uploaded_pdf in uploaded_pdfs:
            pdf_bytes = uploaded_pdf.read()
            pdf_filename = uploaded_pdf.name.rsplit(".", 1)[0]
            with st.spinner(f"⏳ 分析中：{uploaded_pdf.name}"):
                # ✅ 自動判斷是否為掃描 PDF
                full_text = extract_text_by_line(pdf_bytes)
                facility_prompt = FACILITY_PROMPT.replace("{text}", full_text)
                facility_type = get_gemini_response(facility_prompt).strip()
                plan_prompt = PLAN_PROMPT.replace("{text}", full_text)
                plan_info_result = get_gemini_response(plan_prompt)
                legal_compliance_result = query_rag_for_legal_compliance(full_text)
                
                st.info(f"🏥 檔案：{uploaded_pdf.name} → 機構分類：{facility_type}")
                if use_split:
                    prompts = build_individual_prompts(INFO_QUESTIONS, full_text, "資訊處審查委員")
                    responses = [get_gemini_response(p) for p in prompts]
                    info_result = "\n".join(responses)
                    subsidy_prompt = REPEATED_SUBSIDY_PROMPT.replace("{text}", full_text)
                    repeated_subsidy_result = get_gemini_response(subsidy_prompt)
                else:
                    prompt = build_individual_prompts(INFO_QUESTIONS, full_text)[0].replace(
                        "題目：" + INFO_QUESTIONS[0] + "\n\n",
                        "請依照以下題目順序逐一回答每一題：\n\n" + "\n".join(INFO_QUESTIONS)
                    )
                    info_result = get_gemini_response(prompt)
                    subsidy_prompt = REPEATED_SUBSIDY_PROMPT.replace("{text}", full_text)
                    repeated_subsidy_result = get_gemini_response(subsidy_prompt)
            st.success(f"✅ {uploaded_pdf.name} 分析完成")
            render_score_table(
    "💻 資訊處評分表",
    info_result,
    INFO_QUESTIONS,
    "#1f77b4",
    facility_type,
    FACILITY_SCORE_CAP_INFO,
    pdf_filename,
    repeated_subsidy_result,
    plan_info_result,legal_compliance_result
)

if __name__ == "__main__":
    main()
