from PIL import Image, ImageDraw, ImageFont
from textwrap import wrap

BG_PATH = "/mnt/disk1/szchen/VLMBenchmark/repo/Jailbreak-Redteam/attacks_strategy/socialmedia/slack.png"
OUT_PATH = "/mnt/disk1/szchen/VLMBenchmark/repo/Jailbreak-Redteam/attacks_strategy/socialmedia/slack_rendered.png"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

img = Image.open(BG_PATH).convert("RGBA")
draw = ImageDraw.Draw(img)

# =========================
# 字体
# =========================
font_channel = ImageFont.truetype(FONT_PATH, 28)
font_name = ImageFont.truetype(FONT_PATH, 22)
font_time = ImageFont.truetype(FONT_PATH, 16)
font_body = ImageFont.truetype(FONT_PATH, 21)
font_input = ImageFont.truetype(FONT_PATH, 20)

# =========================
# 颜色
# =========================
COLOR_CHANNEL = (35, 39, 42, 255)
COLOR_NAME = (29, 28, 29, 255)
COLOR_TIME = (120, 124, 126, 255)
COLOR_BODY = (29, 28, 29, 255)
COLOR_PLACEHOLDER = (120, 124, 126, 255)
COLOR_AVATAR_TEXT = (255, 255, 255, 255)

# =========================
# 关键区域坐标（按这张图估计）
# =========================
# 主聊天区域（不含左侧栏）
content_left = 430
content_right = 1410

# 顶部频道标题
channel_x = 470
channel_y = 72

# 消息起始区域
msg_left = 470
msg_top = 155
msg_max_width = 870

# 输入框占位文字
input_x = 450
input_y = 835

# =========================
# 顶部频道名
# =========================
channel_name = "# general"
draw.text((channel_x, channel_y), channel_name, font=font_channel, fill=COLOR_CHANNEL)

# =========================
# Slack 消息数据
# =========================
messages = [
    {
        "name": "Alice",
        "time": "10:12 AM",
        "avatar_color": (54, 197, 240, 255),
        "avatar_text": "A",
        "text": "Morning everyone. I just uploaded the latest experiment results and summarized the main findings in the doc."
    },
    {
        "name": "Brian",
        "time": "10:14 AM",
        "avatar_color": (46, 182, 125, 255),
        "avatar_text": "B",
        "text": "Nice. I checked the charts and the trend looks much better than last week. We should probably highlight the ablation section too."
    },
    {
        "name": "Cindy",
        "time": "10:16 AM",
        "avatar_color": (230, 92, 92, 255),
        "avatar_text": "C",
        "text": "Agreed. I can prepare a short summary for the meeting this afternoon if that helps."
    },
]

# =========================
# 工具函数
# =========================
def draw_avatar(draw, x, y, size, fill, text, font):
    draw.rounded_rectangle((x, y, x + size, y + size), radius=10, fill=fill)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = x + (size - tw) / 2
    ty = y + (size - th) / 2 - 2
    draw.text((tx, ty), text, font=font, fill=COLOR_AVATAR_TEXT)

def wrap_text_pixel(draw, text, font, max_width):
    words = text.split()
    lines = []
    cur = ""
    for word in words:
        test = word if not cur else cur + " " + word
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines

# =========================
# 渲染消息
# =========================
avatar_size = 38
avatar_font = ImageFont.truetype(FONT_PATH, 20)
name_x_offset = 56
body_x_offset = 56
msg_gap = 30
line_spacing = 8

y = msg_top

for m in messages:
    avatar_x = msg_left
    avatar_y = y + 2
    draw_avatar(draw, avatar_x, avatar_y, avatar_size, m["avatar_color"], m["avatar_text"], avatar_font)

    # 姓名
    name_x = msg_left + name_x_offset
    name_y = y
    draw.text((name_x, name_y), m["name"], font=font_name, fill=COLOR_NAME)

    # 时间
    name_bbox = draw.textbbox((name_x, name_y), m["name"], font=font_name)
    time_x = name_bbox[2] + 10
    time_y = y + 4
    draw.text((time_x, time_y), m["time"], font=font_time, fill=COLOR_TIME)

    # 正文
    body_x = msg_left + body_x_offset
    body_y = y + 30
    body_lines = wrap_text_pixel(draw, m["text"], font_body, msg_max_width - body_x_offset)

    line_bbox = draw.textbbox((0, 0), "Ag", font=font_body)
    line_h = (line_bbox[3] - line_bbox[1]) + line_spacing

    for line in body_lines:
        draw.text((body_x, body_y), line, font=font_body, fill=COLOR_BODY)
        body_y += line_h

    y = body_y + msg_gap

# =========================
# 输入框占位文字
# =========================
draw.text((input_x, input_y), "Message #general", font=font_input, fill=COLOR_PLACEHOLDER)

img.save(OUT_PATH)
print(f"Saved to: {OUT_PATH}")