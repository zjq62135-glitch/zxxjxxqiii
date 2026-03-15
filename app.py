import streamlit as st
import librosa
import numpy as np
import plotly.graph_objects as go

# --- 页面全局配置 ---
st.set_page_config(page_title="课堂音频信息熵分析", layout="wide")
st.title("📈 课堂音频信息熵 (Spectral Entropy) 分析系统")
st.markdown("基于信息论的课堂声学特征提取，直接量化课堂信息输出的复杂度与动态起伏。")

@st.cache_data
def process_audio_entropy(file_bytes):
    """处理音频文件，纯粹提取信息熵及四大统计指标"""
    # 1. 加载音频 
    y, sr = librosa.load(file_bytes, sr=16000)
    
    # 2. 计算短时傅里叶变换 (STFT) 得到频谱图
    S, phase = librosa.magphase(librosa.stft(y))
    power_spectrum = np.abs(S)**2
    
    # 3. 将频谱能量归一化为概率分布 (Probability Mass Function)
    sum_power = np.sum(power_spectrum, axis=0, keepdims=True)
    sum_power[sum_power == 0] = np.finfo(float).eps # 处理绝对静音
    
    prob_spectrum = power_spectrum / sum_power
    prob_spectrum = np.where(prob_spectrum <= 0, np.finfo(float).eps, prob_spectrum) # 防止 log(0)
    
    # 4. 计算香农信息熵 (Spectral Entropy)
    spectral_entropy = -np.sum(prob_spectrum * np.log2(prob_spectrum), axis=0)
    spectral_entropy = np.nan_to_num(spectral_entropy) # 清除潜在的 NaN
    
    # 5. 滑动窗口平滑处理 
    smooth_entropy = np.convolve(spectral_entropy, np.ones(15)/15, mode='same')
    times = librosa.frames_to_time(np.arange(len(smooth_entropy)), sr=sr)
    
    # --- 计算导师和组员要求的 4 个核心指标 ---
    
    max_ent = np.max(smooth_entropy)
    mean_ent = np.mean(smooth_entropy)
    
    # 波峰因数 (Crest Factor)：听取之前的建议，这里采用 95% 分位数代替绝对最大值，防噪点
    h_95 = np.percentile(smooth_entropy, 95)
    crest_factor = h_95 / mean_ent if mean_ent > 0 else 0
    
    # 定义高低唤醒度阈值 (75% 和 25% 分位数)
    threshold_peak = np.percentile(smooth_entropy, 75)   
    threshold_valley = np.percentile(smooth_entropy, 25) 
    
    # 信息低谷比例
    valley_ratio = np.sum(smooth_entropy < threshold_valley) / len(smooth_entropy)
    
    return times, smooth_entropy, max_ent, mean_ent, crest_factor, valley_ratio, threshold_peak, threshold_valley

# --- 网页 UI 与交互逻辑 ---
uploaded_file = st.file_uploader("📂 请上传课堂音频文件 (.wav / .mp3)", type=['wav', 'mp3'])

if uploaded_file is not None:
    with st.spinner("⏳ 正在进行短时傅里叶变换与信息熵计算，请稍候..."):
        # 执行算法
        (times, smooth_entropy, max_ent, mean_ent, 
         crest_factor, valley_ratio, t_peak, t_valley) = process_audio_entropy(uploaded_file)
        
    st.success("✅ 信息熵特征提取完成！")
    
    # --- 模块一：四大核心指标看板 ---
    st.markdown("### 📊 课堂教学节奏质量诊断")
    col1, col2, col3, col4 = st.columns(4)
    
    # 调整了顺序，将平均和最高放在前面，波峰因数和低谷比放在后面
    col1.metric("平均信息熵 (Mean)", f"{mean_ent:.2f} bits", "整体信息密度")
    col2.metric("最高信息熵 (Max)", f"{max_ent:.2f} bits", "绝对峰值复杂度")
    col3.metric("波峰因数 (Crest Factor)", f"{crest_factor:.2f}", "教学设计紧凑性/爆发力")
    col4.metric("信息低谷比 (Valley Ratio)", f"{valley_ratio * 100:.1f} %", "低唤醒度/单调沉闷占比", delta_color="inverse")

    # --- 模块二：信息熵动态波形图 ---
    st.markdown("### 📈 课堂信息熵时间序列波形图")
    
    # 降采样渲染以保证网页流畅度
    downsample_rate = max(1, len(times) // 3000)
    plot_times = times[::downsample_rate]
    plot_entropy = smooth_entropy[::downsample_rate]
    
    fig = go.Figure()
    
    # 绘制主体信息熵曲线
    fig.add_trace(go.Scatter(
        x=plot_times, y=plot_entropy,
        mode='lines',
        name='信息熵 (Spectral Entropy)',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # 添加“高唤醒度基准线” (上四分位数, 75%)
    fig.add_hline(y=t_peak, line_dash="dash", line_color="red", 
                  annotation_text="高唤醒阈值 (75th Percentile)", annotation_position="top left")
    
    # 添加“低唤醒度基准线” (下四分位数, 25%)
    fig.add_hline(y=t_valley, line_dash="dash", line_color="green", 
                  annotation_text="低唤醒阈值 (25th Percentile)", annotation_position="bottom left")

    # 优化图表样式
    fig.update_layout(
        xaxis_title="时间 (秒)",
        yaxis_title="信息熵值 (bits)",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor='rgba(240, 242, 246, 0.5)',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **💡 核心图表与指标解读：**
    * **高/低唤醒度区间**：图中红线以上（>75%）为课堂的高频互动或激昂期（高唤醒值）；绿线以下（<25%）为平铺直叙或静默期（低唤醒度）。
    * **波峰因数 (Crest Factor)**：使用 $H_{95} / H_{avg}$ 计算，排除了极端异常噪音。该值越大，说明课堂高潮与平均水平的落差越明显，教学节奏的“推弦感”和紧凑性越强。
    * **信息低谷比**：反映了一节课中缺乏新异刺激、容易导致学生注意力涣散的时间占比。
    """)