import streamlit as st
import io
import os
import random
import numpy as np
import pdfplumber
import matplotlib

matplotlib.use('Agg')  # 服务器端绘图必须设置
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Image

# --- 配置 ---
# 这里的字体路径必须是相对路径，因为要上传到服务器
FONT_PATH = "simsun.ttc"
FONT_NAME = "SimSun"


# --- 核心逻辑函数 (复用之前的，稍作修改) ---

def register_font():
    """注册字体，如果本地没有simsun，会尝试报错提示"""
    if os.path.exists(FONT_PATH):
        pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_PATH))
        return True
    else:
        st.error(f"❌ 错误：在项目根目录下找不到字体文件 '{FONT_PATH}'。请务必将字体文件上传到GitHub项目库中！")
        return False


def extract_text_from_upload(uploaded_file):
    """从上传的内存文件中提取文本"""
    content = []
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    lines = [line.strip() for line in text.split('\n') if line.strip()]
                    content.extend(lines)
    except Exception as e:
        st.error(f"解析PDF失败: {e}")
    return content


# ... (此处保留 create_academic_chart 和 create_math_formula 函数，代码完全不用变) ...
# 为了节省篇幅，这里假设你已经把上一轮回复中的这两个绘图函数复制过来了
# 务必把 create_academic_chart 和 create_math_formula 完整粘贴在这里
# -----------------------------------------------------------------
def create_academic_chart():
    # ... (粘贴上一段代码中的实现) ...
    plt.style.use('ggplot')
    chart_type = random.choice(['load', 'convergence', 'voltage'])
    fig, ax = plt.subplots(figsize=(5, 3.5))
    font_dict = {'family': 'serif', 'size': 10}
    if chart_type == 'load':
        t = np.arange(0, 24, 0.5)
        load = 60 + 20 * np.sin((t - 6) * np.pi / 12) ** 2 + np.random.normal(0, 2, len(t))
        ax.plot(t, load, 'k-', linewidth=1)
        ax.fill_between(t, load, alpha=0.3, color='gray')
        ax.set_xlabel('Time (h)', fontdict=font_dict)
        ax.set_ylabel('Active Power (MW)', fontdict=font_dict)
        ax.set_title('Fig. Daily Load Profile simulation.', fontdict={'family': 'serif', 'size': 9, 'weight': 'bold'})
    elif chart_type == 'convergence':
        episodes = np.arange(100)
        reward = -np.exp(-0.05 * episodes) + 0.1 * np.random.rand(100)
        ax.plot(episodes, reward, color='#1f77b4', label='Proposed')
        ax.plot(episodes, reward - 0.2, color='#ff7f0e', linestyle='--', label='Baseline')
        ax.legend(prop={'family': 'serif', 'size': 8})
        ax.set_xlabel('Episodes', fontdict=font_dict)
        ax.set_ylabel('Average Reward', fontdict=font_dict)
        ax.set_title('Fig. Training convergence analysis.', fontdict={'family': 'serif', 'size': 9, 'weight': 'bold'})
    else:
        nodes = np.arange(1, 15)
        v = 1.0 - 0.08 * np.random.rand(14)
        ax.bar(nodes, v, color='#2ca02c', alpha=0.7)
        ax.axhline(0.95, color='r', linestyle=':', linewidth=1)
        ax.set_ylim(0.85, 1.05)
        ax.set_xlabel('Bus Index', fontdict=font_dict)
        ax.set_ylabel('Voltage (p.u.)', fontdict=font_dict)
        ax.set_title('Fig. Node voltage distribution.', fontdict={'family': 'serif', 'size': 9, 'weight': 'bold'})
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=200)
    plt.close()
    buf.seek(0)
    return buf


def create_math_formula():
    formulas = [
        r'$ J = \sum_{t=1}^{T} (C_{loss} P_{loss,t} + C_{sw} N_{sw,t}) $',
        r'$ P_{i} = U_i \sum_{j \in \Omega_i} U_j (G_{ij} \cos \theta_{ij} + B_{ij} \sin \theta_{ij}) $',
        r'$ Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a} Q(s,a) - Q(s,a)] $',
        r'$ \min \sum_{i \in \mathcal{N}} (P_{Gi} - P_{Di})^2 $'
    ]
    formula = random.choice(formulas)
    fig = plt.figure(figsize=(4, 0.8))
    fig.text(0.5, 0.5, formula, size=16, ha='center', va='center', family='serif')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=200, transparent=True, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf


def header_footer_template(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.black)
    canvas.setFillColor(colors.black)
    canvas.setFont('Times-Roman', 9)
    header_text = "IEEE TRANSACTIONS ON POWER SYSTEMS, VOL. 41, NO. 1, JANUARY 2026"
    canvas.drawString(0.8 * inch, 11.2 * inch, header_text)
    canvas.drawString(7.5 * inch, 11.2 * inch, str(canvas.getPageNumber()))
    canvas.line(0.8 * inch, 11.1 * inch, 7.7 * inch, 11.1 * inch)
    canvas.restoreState()


def generate_pdf(novel_lines, title_text):
    """主生成逻辑，返回 BytesIO 对象"""
    buffer = io.BytesIO()
    doc = BaseDocTemplate(buffer, pagesize=A4,
                          leftMargin=0.8 * inch, rightMargin=0.8 * inch,
                          topMargin=0.8 * inch, bottomMargin=0.8 * inch)

    col_width = 3.25 * inch
    gutter = 0.2 * inch
    frame1 = Frame(doc.leftMargin, doc.bottomMargin, col_width, doc.height, id='col1')
    frame2 = Frame(doc.leftMargin + col_width + gutter, doc.bottomMargin, col_width, doc.height, id='col2')
    doc.addPageTemplates([PageTemplate(id='TwoCol', frames=[frame1, frame2], onPage=header_footer_template)])

    styles = getSampleStyleSheet()
    # 重新定义样式，确保字体正确
    title_style = ParagraphStyle(name='PaperTitle', parent=styles['Heading1'], fontName='Times-Bold', fontSize=18,
                                 leading=22, alignment=TA_CENTER, spaceAfter=12)
    abstract_style = ParagraphStyle(name='Abstract', fontName='Times-Bold', fontSize=9, leading=11,
                                    alignment=TA_JUSTIFY, spaceAfter=10)
    body_style = ParagraphStyle(name='Body', fontName=FONT_NAME, fontSize=10, leading=14, alignment=TA_JUSTIFY,
                                firstLineIndent=20, spaceAfter=5)

    story = []
    # 标题
    story.append(Paragraph(title_text, title_style))
    story.append(Spacer(1, 10))
    # 摘要
    abstract_text = "<b>Abstract—</b> " + (novel_lines[0][:300] if len(novel_lines) > 0 else "Analysis...") + "..."
    story.append(Paragraph(abstract_text, abstract_style))
    story.append(Spacer(1, 15))

    para_count = 0
    img_prob = 0.05

    # 进度条
    progress_bar = st.progress(0)
    total_lines = len(novel_lines)

    for i, line in enumerate(novel_lines):
        if i % 50 == 0:  # 更新进度条
            progress_bar.progress(min(i / total_lines, 1.0))

        if len(line) < 2: continue
        story.append(Paragraph(line, body_style))
        para_count += 1

        # 插入图片
        if para_count > 10 and random.random() < img_prob:
            img = Image(create_academic_chart())
            img_width = col_width - 10
            aspect = img.imageHeight / float(img.imageWidth)
            img.drawWidth = img_width
            img.drawHeight = img_width * aspect
            story.append(Spacer(1, 6));
            story.append(img);
            story.append(Spacer(1, 6))
            img_prob = 0.01
        else:
            img_prob = min(0.06, img_prob + 0.005)

        # 插入公式
        if para_count > 5 and random.random() < 0.08:
            img = Image(create_math_formula())
            img_height = 0.4 * inch
            aspect = img.imageHeight / float(img.imageWidth)
            img.drawHeight = img_height
            img.drawWidth = img_height / aspect
            story.append(Spacer(1, 4));
            story.append(img);
            story.append(Spacer(1, 4))

    doc.build(story)
    buffer.seek(0)
    return buffer


# --- Streamlit 界面 ---

st.set_page_config(page_title="研究生摸鱼神器", layout="centered")

st.title("⚡ 研究生论文伪装器 (电力版)")
st.write("上传小说 PDF，自动转化为 IEEE Transactions 格式，内含电力系统仿真图表。")

# 侧边栏配置
with st.sidebar:
    st.header("论文设置")
    paper_title = st.text_input("论文标题 (英文)",
                                value="A Distributed Dynamic Reconfiguration Strategy for Resilient Power Systems")
    st.info("提示：请确保目录下包含 simsun.ttc 字体文件")

# 1. 注册字体
if not register_font():
    st.stop()

# 2. 文件上传
uploaded_file = st.file_uploader("请上传小说 PDF 文件 (纯文字版)", type=["pdf"])

if uploaded_file is not None:
    if st.button("开始伪装 / Generate Paper"):
        with st.spinner('正在提取文本并进行仿真实验 (生成中)...'):
            # 提取文本
            lines = extract_text_from_upload(uploaded_file)

            if len(lines) > 0:
                # 生成 PDF
                pdf_buffer = generate_pdf(lines, paper_title)

                st.success("✅ 伪装成功！点击下方按钮下载。")

                # 下载按钮
                st.download_button(
                    label="📥 下载摸鱼论文 (PDF)",
                    data=pdf_buffer,
                    file_name="research_paper_v1.pdf",
                    mime="application/pdf"
                )
            else:
                st.error("未能提取到文本，请检查PDF是否为扫描件。")