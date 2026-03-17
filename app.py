import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import whisper
import jieba
import math
from collections import Counter
import plotly.graph_objects as go
import os
import numpy as np

# --- 强制修改背景颜色为白色 ---
st.markdown(
    """
    <style>
    /* 主内容区域背景 */
    .stApp {
        background-color: white;
        color: black; /* 强制所有文字变成黑色！ */
    }
    /* 顶部页眉背景 */
    header[data-testid="stHeader"] {
        background-color: white;
    }
    /* 针对一些组件背景的微调 (可选) */
    .stMetric {
        background-color: #f8f9fa; /* 给数据指标卡片加一点淡淡的灰，方便识别 */
        padding: 10px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- 页面全局配置 ---
st.set_page_config(page_title="课堂语义信息熵分析", layout="wide")
st.title("教学质量检测")
st.markdown("通过本地部署的 OpenAI Whisper 模型提取教师授课文本，结合中文分词技术，计算文本的香农信息熵")

# --- 核心模型加载 (使用缓存，防止每次上传都重新加载模型) ---
@st.cache_resource
def load_whisper_model():
    # 默认使用 "base" 模型（占用内存小，速度快，适合无 GPU 的普通电脑）。
    # 如果你的电脑有显卡或想要极高准确率，可以改为 "small" 或 "medium"。
    return whisper.load_model("base")

@st.cache_data
def process_audio_to_text_entropy(file_path):
    """
    使用 Whisper 直接处理完整音频，提取带时间戳的语义片段，并计算信息熵。
    """
    model = load_whisper_model()
    
    # Whisper 的神仙功能：直接读取完整音频，自动切分句子，并返回精确时间戳！
    # 第一次运行某个模型时，后台会自动下载模型权重文件（约 140MB），请稍等片刻。
    result = model.transcribe(file_path, language="zh")
    segments = result["segments"]
    
    times = []
    text_entropies = []
    transcripts = []
    
    # 遍历 Whisper 识别出的每一个语音片段
    for seg in segments:
        text = seg["text"].strip()
        start_time = seg["start"]
        end_time = seg["end"]
        
        # 取这句话的中间时刻作为波形图的 X 轴
        mid_time = (start_time + end_time) / 2.0 
        
        # --- NLP 分词与香农熵计算 ---
        if text == "":
            entropy = 0.0
        else:
            # 精确模式分词
            words = list(jieba.cut(text, cut_all=False))
            # 过滤标点符号和常见的无意义语气词 (停用词)
            stop_words = ["，", "。", "！", "？", "、", "：", " ", "的", "了", "是", "啊", "呃", "这个", "那个"]
            words = [w for w in words if w.strip() and w not in stop_words]
            
            if not words:
                entropy = 0.0
            else:
                total_words = len(words)
                word_counts = Counter(words)
                # 计算文本信息熵 H = - sum( p * log2(p) )
                entropy = -sum((count / total_words) * math.log2(count / total_words) for count in word_counts.values())
        
        times.append(mid_time)
        text_entropies.append(entropy)
        transcripts.append(text)
        
    # --- 计算全局核心指标 ---
    valid_entropies = [e for e in text_entropies if e > 0]
    
    if not valid_entropies:
        return times, text_entropies, 0, 0, 0, 0, 0, 0, transcripts
        
    max_ent = float(np.max(valid_entropies))
    mean_ent = float(np.mean(valid_entropies))
    
    h_95 = float(np.percentile(valid_entropies, 95))
    crest_factor = float(h_95 / mean_ent) if mean_ent > 0 else 0
    
    threshold_peak = float(np.percentile(valid_entropies, 75))
    threshold_valley = float(np.percentile(valid_entropies, 25))
    
    valley_ratio = float(sum(1 for e in valid_entropies if e < threshold_valley) / len(valid_entropies))
    
    return times, text_entropies, max_ent, mean_ent, crest_factor, valley_ratio, threshold_peak, threshold_valley, transcripts

# --- 网页 UI 与交互逻辑 ---
uploaded_file = st.file_uploader("📂 请上传课堂音频文件 (支持 .wav, .mp3, .m4a 等格式)", 
                                 type=['wav', 'mp3', 'm4a', 'aac', 'flac'])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    temp_audio_path = f"uploaded_audio.{file_extension}"
    
    with open(temp_audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    with st.spinner(f"⏳ 正在调用本地大模型进行深度语义解析... (如遇初次加载模型可能需要 1-2 分钟，断网亦可运行)"):
        (times, text_entropies, max_ent, mean_ent, 
         crest_factor, valley_ratio, t_peak, t_valley, transcripts) = process_audio_to_text_entropy(temp_audio_path)
        
    st.success("✅ 本地模型识别与语义信息熵计算完成！")
    
    # --- 模块一：核心指标数据看板 ---
    st.markdown("### 📊 课堂内容干货密度诊断")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("平均语义信息熵", f"{mean_ent:.2f} bits", "平均词汇丰富度")
    col2.metric("最高语义信息熵", f"{max_ent:.2f} bits", "知识最密集瞬间")
    col3.metric("知识波峰因数", f"{crest_factor:.2f}", "内容深度的爆发力")
    col4.metric("内容低谷比例", f"{valley_ratio * 100:.1f} %", "水话/重复话占比", delta_color="inverse")

    # --- 模块二：语义信息熵动态波形图 ---
    st.markdown("### 📈 授课内容语义信息熵时间序列图")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=times, y=text_entropies,
        mode='lines+markers',
        name='语义信息熵',
        line=dict(color='#8A2BE2', width=2),
        marker=dict(size=6),
        text=transcripts,
        hovertemplate="时间: 第 %{x:.1f} 秒<br>干货密度 (熵): %{y:.2f} bits<br>识别原文: %{text}<extra></extra>"
    ))
    
    fig.add_hline(y=t_peak, line_dash="dash", line_color="red", 
                  annotation_text="高干货阈值 (75%)", annotation_position="top left")
    
    fig.add_hline(y=t_valley, line_dash="dash", line_color="green", 
                  annotation_text="低信息阈值 (25%)", annotation_position="bottom left")

    fig.update_layout(
        xaxis_title="课程进行时间 (秒)",
        yaxis_title="语义信息熵值 (bits)",
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor='rgba(240, 242, 246, 0.5)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **💡 Whisper 语义模型图表解读：**
    * 🟣 **紫色曲线**：代表老师每句话的“干货程度”。**鼠标悬停在点上，可以直接查看原话！**
    * 📈 **高干货区（红线以上）**：这段时间老师输出了大量不重复的专业术语，知识密度极高。
    * 📉 **低信息区（绿线以下）**：可能在说一些过渡性的话语、重复的语气词（如：啊、呃、这个），或者是在等待学生回应。
    """)
    
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)
