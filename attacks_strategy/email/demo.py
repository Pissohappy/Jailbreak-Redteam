from PIL import Image, ImageDraw, ImageFont
import textwrap

# =========================
# 1. 路径设置
# =========================
BG_PATH = "/mnt/disk1/szchen/VLMBenchmark/repo/Jailbreak-Redteam/attacks_strategy/email/email3.png"
OUT_PATH = "/mnt/disk1/szchen/VLMBenchmark/repo/Jailbreak-Redteam/attacks_strategy/email/email3_rendered.png"

# 改成你机器上的字体路径
# Windows 示例:
# FONT_PATH = "C:/Windows/Fonts/arial.ttf"
# Mac 示例:
# FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"
# Linux 示例:
# FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# =========================
# 2. 要渲染的内容
# =========================
from_text = "Alice <alice@example.com>"
to_text = "Bob <bob@example.com>"
cc_text = "Charlie <charlie@example.com>"
subject_text = "Project Update for This Week"

body_text = """Hi Bob,

Here is the latest update on the project.

1. The UI draft has been completed.
2. The backend API is under integration.
3. We expect the internal review by Friday.

Please let me know if you want a separate summary deck.

Best,
Alice
"""

# =========================
# 3. 载入图片和字体
# =========================
img = Image.open(BG_PATH).convert("RGBA")
draw = ImageDraw.Draw(img)

# 以下是email1.png的设置，字体较大一些
font_label = ImageFont.truetype(FONT_PATH, 26)     # From/To/Cc/Subject 标签
font_text = ImageFont.truetype(FONT_PATH, 24)      # 内容文字
font_subject = ImageFont.truetype(FONT_PATH, 24)   # Subject 内容
font_body = ImageFont.truetype(FONT_PATH, 24)      # 正文

# 以下是email2.png和email3.png的设置，字体更小一些
font_text = ImageFont.truetype(FONT_PATH, 15)
font_subject = ImageFont.truetype(FONT_PATH, 15)
font_body = ImageFont.truetype(FONT_PATH, 14)

# =========================
# 4. 颜色
# =========================
TEXT_COLOR = (30, 30, 30, 255)
SUBJECT_COLOR = (30, 30, 30, 255)
BODY_COLOR = (40, 40, 40, 255)

# =========================
# 5. 坐标（基于这张 1536x1024 图手动估计）
#    你可以微调这些值
# =========================

# 下列参数是针对email1.png设计
x_label = 100
x_value = 220

y_from = 135
y_to = 195
y_cc = 255
y_subject = 320

body_left = 100
body_top = 410
body_right = 1430
line_spacing = 14

# 下列参数是针对email2.png设计

x_value = 425

y_from = 210
y_to = 250
y_cc = 290
y_subject = 315

# 正文区域
body_left = 347
body_top = 350
body_right = 1147
line_spacing = 8

# 下列参数是针对email3.png设计

x_value = 420

y_from = 210
y_to = 250
y_cc = 290
y_subject = 328

# 正文区域
body_left = 347
body_top = 360
body_right = 1147
line_spacing = 8



# =========================
# 6. 画头部字段
# =========================
draw.text((x_value, y_from), from_text, font=font_text, fill=TEXT_COLOR)
draw.text((x_value, y_to), to_text, font=font_text, fill=TEXT_COLOR)
draw.text((x_value, y_cc), cc_text, font=font_text, fill=TEXT_COLOR)
draw.text((x_value, y_subject), subject_text, font=font_subject, fill=SUBJECT_COLOR)

# =========================
# 7. 正文自动换行
# =========================
def wrap_text_by_width(text, font, max_width, draw):
    lines = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue

        words = paragraph.split(" ")
        current = ""
        for word in words:
            test_line = word if current == "" else current + " " + word
            bbox = draw.textbbox((0, 0), test_line, font=font)
            w = bbox[2] - bbox[0]
            if w <= max_width:
                current = test_line
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines

max_body_width = body_right - body_left
body_lines = wrap_text_by_width(body_text, font_body, max_body_width, draw)

# 获取单行高度
sample_bbox = draw.textbbox((0, 0), "Ag", font=font_body)
line_height = (sample_bbox[3] - sample_bbox[1]) + line_spacing

y = body_top
for line in body_lines:
    draw.text((body_left, y), line, font=font_body, fill=BODY_COLOR)
    y += line_height

# =========================
# 8. 保存
# =========================
img.save(OUT_PATH)
print(f"Saved to: {OUT_PATH}")