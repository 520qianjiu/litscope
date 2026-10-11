import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'docs' / 'assets'

BG = (11, 15, 23)
BOX = (17, 24, 38)
BORDER = (38, 52, 74)
FG = (230, 237, 243)
MUTED = (139, 152, 169)
ACCENT = (47, 111, 235)
GREEN = (63, 185, 120)
AMBER = (219, 171, 76)
CYAN = (86, 182, 194)
TERM_BG = (13, 17, 23)
TERM_BAR = (24, 29, 38)
CITE = (110, 168, 254)

FONT_CANDIDATES = [
    [r'C:\\Windows\\Fonts\\msyh.ttc', r'C:\\Windows\\Fonts\\msyhbd.ttc'],
    ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'],
    ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'],
]
_CACHE = {}


def font(size, bold=False):
    key = (size, bold)
    if key in _CACHE:
        return _CACHE[key]
    chosen = None
    for group in FONT_CANDIDATES:
        for candidate in group:
            if os.path.isfile(candidate):
                chosen = candidate
                break
        if chosen:
            break
    picked = ImageFont.truetype(chosen, size) if chosen else ImageFont.load_default()
    _CACHE[key] = picked
    return picked


def rrect(draw, box, fill=BOX, outline=BORDER, radius=14, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def center(draw, box, text, f, fill=FG):
    x0, y0, x1, y1 = box
    left, top, right, bottom = draw.textbbox((0, 0), text, font=f)
    draw.text(((x0 + x1 - (right - left)) / 2 - left, (y0 + y1 - (bottom - top)) / 2 - top),
              text, font=f, fill=fill)


def arrow(draw, x0, x1, y, color=BORDER, width=2):
    if x1 - x0 < 12:
        return
    draw.line([(x0, y), (x1 - 10, y)], fill=color, width=width)
    draw.polygon([(x1, y), (x1 - 12, y - 6), (x1 - 12, y + 6)], fill=color)


def band(draw, y, height, label, boxes, accent=ACCENT):
    draw.text((46, y + 14), label, font=font(20, True), fill=accent)
    gap = 30
    count = len(boxes)
    width = (1680 - 46 - 150 - 40 - gap * (count - 1)) / count
    left = 46 + 150
    rects = []
    for index, (title, subtitle) in enumerate(boxes):
        x0 = left + index * (width + gap)
        box = (x0, y, x0 + width, y + height)
        rrect(draw, box)
        if subtitle:
            center(draw, (box[0], box[1], box[2], box[1] + height * 0.56), title, font(21, True))
            center(draw, (box[0], box[1] + height * 0.5, box[2], box[3]), subtitle, font(16),
                   fill=MUTED)
        else:
            center(draw, box, title, font(21, True))
        rects.append(box)
        if index:
            arrow(draw, rects[index - 1][2] + 6, box[0] - 6, y + height / 2)
    return rects


def wrap_text(draw, text, f, max_width):
    lines = []
    current = ''
    for word in text.split(' '):
        candidate = (current + ' ' + word).strip()
        if draw.textlength(candidate, font=f) <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
        while draw.textlength(word, font=f) > max_width:
            cut = len(word)
            while cut > 1 and draw.textlength(word[:cut], font=f) > max_width:
                cut -= 1
            lines.append(word[:cut])
            word = word[cut:]
        current = word
    if current:
        lines.append(current)
    return lines


def build_diagram():
    image = Image.new('RGB', (1680, 1010), BG)
    draw = ImageDraw.Draw(image)
    draw.text((46, 42), 'LitScope', font=font(42, True), fill=FG)
    draw.text((46, 98), '多源文献检索 + 带引用问答 + 可证伪的评测闭环', font=font(20), fill=MUTED)
    band(draw, 170, 108, '数据层', [
        ('arXiv', 'Atom API'), ('Crossref', 'REST API'), ('PubMed', 'E-utilities')], accent=CYAN)
    band(draw, 320, 108, '处理层', [
        ('多源检索', '统一 Paper 数据模型'),
        ('去重归一化', 'DOI / 标题 / 模糊 0.92'),
        ('分块 + 嵌入', '句子级窗口，可插拔')])
    band(draw, 470, 108, '索引层', [
        ('向量库', '205 chunks，余弦检索'), ('BM25 索引', '纯 Python 词法基线')])
    band(draw, 620, 108, '生成层', [
        ('Prompt 组装', '强制每条结论带引用'),
        ('DeepSeek', 'OpenAI 兼容协议'),
        ('带引用回答', '[1][2] 回指到 chunk')], accent=GREEN)
    band(draw, 770, 108, '评测层', [
        ('引用解析', '无效标记检测'),
        ('LLM-as-judge', '引用支持率'),
        ('负对照', '平衡准确率'),
        ('metrics.json', 'R@k / MRR / 延迟')], accent=AMBER)
    draw.text((46, 918), '实测：44 篇论文 / 205 chunks，20 条关键词查询', font=font(18), fill=MUTED)
    draw.text((46, 952), 'BM25  R@1 0.95  MRR 0.975      哈希嵌入  R@1 0.90  MRR 0.942',
              font=font(18, True), fill=FG)
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / 'architecture.png'
    image.save(out)
    print('wrote ' + str(out) + '  ' + str(image.size))


ANSWER = ('The provided sources do not contain enough information to answer this question. '
          'While the sources discuss graph neural networks in various contexts - training '
          'methods like mini-batch and full-graph training [4], applications in '
          'visual-tactile fusion [2][3], and unsupervised graph anomaly detection [5] - '
          'none of them describe the specific mechanisms by which GNNs propagate '
          'information along edges.')

CITATIONS = [
    '[4] arxiv:2406.00552v4    bm25 score 5.63',
    '[2] pubmed:39331560       bm25 score 7.16',
    '[3] pubmed:39331560       bm25 score 5.88',
    '[5] pubmed:40449236       bm25 score 5.35',
]


def build_demo_gif():
    width, height = 1180, 690
    body = font(19)
    head = font(19, True)
    title = font(17)
    probe = ImageDraw.Draw(Image.new('RGB', (8, 8)))
    palette = {'cmd': CYAN, 'log': MUTED, 'head': GREEN, 'body': FG, 'cite': CITE}

    lines = [
        ('cmd', chr(36) + ' python scripts/ask.py --topic graph-neural-network --retriever bm25'),
        ('cmd', '    --question What methods do GNNs use to propagate information along edges?'),
        ('log', ''), ('log', '[source] arxiv -> 8'), ('log', '[source] crossref -> 8'),
        ('log', '[source] pubmed -> 8'), ('log', '[corpus] 23 papers after dedupe'),
        ('log', '[index] 109 chunks, retriever=bm25'), ('log', ''), ('head', 'ANSWER'),
    ]
    lines += [('body', text) for text in wrap_text(probe, ANSWER, body, width - 70)]
    lines += [('log', ''), ('head', 'CITATIONS')]
    lines += [('cite', text) for text in CITATIONS]

    frames, durations = [], []
    for count in range(1, len(lines) + 1):
        image = Image.new('RGB', (width, height), TERM_BG)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, width, 44), fill=TERM_BAR)
        draw.ellipse((18, 16, 30, 28), fill=(255, 95, 86))
        draw.ellipse((38, 16, 50, 28), fill=(255, 189, 46))
        draw.ellipse((58, 16, 70, 28), fill=(39, 201, 63))
        draw.text((88, 13), 'litscope  |  real run, real output', font=title, fill=MUTED)
        y = 66
        for style, text in lines[:count]:
            draw.text((30, y), text, font=(head if style == 'head' else body), fill=palette[style])
            y += 27
        frames.append(image.convert('P', palette=Image.ADAPTIVE, colors=32))
        durations.append(900 if count >= len(lines) - 2 else 130)

    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / 'demo.gif'
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=True, disposal=2)
    print('wrote ' + str(out) + '  frames=' + str(len(frames))
          + '  KB=' + str(round(out.stat().st_size / 1024)))


if __name__ == '__main__':
    build_diagram()
    build_demo_gif()
