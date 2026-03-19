import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from faster_whisper import WhisperModel
import hanlp  # 引入深度学习 NLP 引擎
import math
from collections import Counter
import plotly.graph_objects as go
import numpy as np

# --- 页面全局配置 ---
st.set_page_config(
    page_title="教学质量检测", 
    page_icon="🎓", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 600;
        color: #636efa;
    }
    .title-section {
        margin-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='title-section'>", unsafe_allow_html=True)
st.title("🎓 教学质量检测")
st.markdown("##### 基于 Faster-Whisper 与 HanLP 深度学习双引擎的课堂内容量化")
st.caption("采用前沿 NLP 神经网络架构与香农信息论，动态提取授课语义特征，为教学复盘提供精准数据支撑。")
st.markdown("</div>", unsafe_allow_html=True)
st.divider()

# --- 核心模型加载区 ---
@st.cache_resource
def load_whisper_model():
    # 语音大模型：极速版
    return WhisperModel("base", device="cpu", compute_type="int8")

@st.cache_resource
def load_hanlp_tokenizer():
    # 加载 HanLP 轻量级预训练分词模型 (ELECTRA 架构)
    # 第一次运行会自动下载模型权重，请耐心等待
    return hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)

@st.cache_data
def process_audio_to_text_entropy(file_path):
    model = load_whisper_model()
    tokenizer = load_hanlp_tokenizer() # 调用 HanLP
    
    segments, info = model.transcribe(file_path, language="zh", beam_size=5)
    
    times, text_entropies, transcripts = [], [], []
    
    for seg in segments:
        text = seg.text.strip()
        mid_time = (seg.start + seg.end) / 2.0 
        
        if text == "":
            entropy = 0.0
        else:
            # 使用 HanLP 神经网络进行高精度分词
            words = tokenizer(text)
            
            # 读取停用词库的容错机制
            try:
                with open('hit_stopwords.txt', 'r', encoding='utf-8') as f:
                    stop_words = f.read().splitlines()
            except FileNotFoundError:
                stop_words = ["，", "。", "！", "？", "、", "：", " ", "的", "了", "是", "啊", "呃", "这个", "那个"]
                
            words = [w for w in words if w.strip() and w not in stop_words]
            
            if not words:
                entropy = 0.0
            else:
                total_words = len(words)
                word_counts = Counter(words)
                entropy = -sum((count / total_words) * math.log2(count / total_words) for count in word_counts.values())
        
        times.append(mid_time)
        text_entropies.append(entropy)
        transcripts.append(text)
        
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

# --- 文件上传区域 (支持 MP4 视频直传) ---
uploaded_file = st.file_uploader(
    "📂 选择课堂音视频文件上传 (支持 .mp4, .wav, .mp3, .m4a 等格式)", 
    type=['wav', 'mp3', 'm4a', 'aac', 'flac', 'mp4']
)

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    temp_audio_path = f"temp_upload.{file_extension}"
    
    with open(temp_audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    # 提供媒体预览
    if file_extension == 'mp4':
        with st.expander("📺 点击预览上传的课堂视频", expanded=False):
            st.video(temp_audio_path)
    else:
        with st.expander("🔊 点击试听上传的课堂音频", expanded=False):
            st.audio(temp_audio_path)
            
    with st.status("🧠 正在启动 AI 双引擎解析...", expanded=True) as status:
        st.write("1. 正在唤醒 Faster-Whisper 语音大模型...")
        st.write("2. 正在提取精准时间戳文本...")
        st.write("3. 正在调用 HanLP 神经网络进行深层语义分词与测算...")
        
        (times, text_entropies, max_ent, mean_ent, 
         crest_factor, valley_ratio, t_peak, t_valley, transcripts) = process_audio_to_text_entropy(temp_audio_path)
         
        status.update(label="✅ 分析报告生成完毕！", state="complete", expanded=False)
    
    st.write("") 
    
    # --- 数据看板 ---
    st.markdown("#### 📊 课堂内容深度核心指标")
    st.info("""提示：
            \n 1. 平均信息熵越高，代表授课词汇越丰富、专业度越强；
            \n 2. 最大信息熵反应整节课最高潮的部分；
            \n 3. 波峰因数越高，老师讲课节奏越快；
            \n 4. 信息低谷比例过高，通常意味着课堂存在较多停顿或无意义语气词。""")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("平均信息熵", f"{mean_ent:.2f} bits")
    with col2:
        st.metric("最大信息熵", f"{max_ent:.2f} bits")
    with col3:
        st.metric("波峰因数", f"{crest_factor:.2f}")
    with col4:
        st.metric("信息低谷比例", f"{valley_ratio * 100:.1f} %", delta_color="inverse")

    st.write("")
    st.divider()
    
    # --- 动态波形图 ---
    st.markdown("#### 📈 授课内容语义时序全景图")
    st.caption("鼠标悬停在紫色曲线上，可直接查看该时间点老师的具体授课原话。")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=times, y=text_entropies,
        mode='lines+markers',
        name='干货密度',
        line=dict(color='#636efa', width=2.5, shape='spline'), 
        marker=dict(size=7, color='white', line=dict(width=2, color='#636efa')), 
        text=transcripts,
        hovertemplate="<b>时间</b>: 第 %{x:.1f} 秒<br><b>干货密度</b>: %{y:.2f} bits<br><br><b>🗣️ 授课原话</b>: <i>%{text}</i><extra></extra>"
    ))
    
    fig.add_hline(y=t_peak, line_dash="dash", line_color="rgba(255, 65, 54, 0.6)", 
                  annotation_text=" 高密度区 (前 25%)", annotation_position="top left", annotation_font_color="rgba(255, 65, 54, 0.6)")
    
    fig.add_hline(y=t_valley, line_dash="dash", line_color="rgba(44, 160, 44, 0.6)", 
                  annotation_text=" 低密度区 (后 25%)", annotation_position="bottom left", annotation_font_color="rgba(44, 160, 44, 0.6)")

    fig.update_layout(
        xaxis_title="课程进行时间 (秒)",
        yaxis_title="语义信息熵 (bits)",
        margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor='rgba(0,0,0,0)', 
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        hoverlabel=dict(bgcolor="white", font_size=13, font_family="Arial") 
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    if os.path.exists(temp_audio_path):
        try:
            os.remove(temp_audio_path)
        except Exception:
            pass
