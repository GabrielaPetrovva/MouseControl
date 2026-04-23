import os
os.environ["PYTHONIOENCODING"] = "utf-8"

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time
import random
import shutil
import json
import requests
from PIL import Image, ImageDraw, ImageFont
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
)
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)

# ──────────────────────────────────────────────
# Шрифт
# ──────────────────────────────────────────────

FONT_PATH = "arial.ttf"

def ensure_font():
    if not os.path.exists(FONT_PATH):
        windows_font = r"C:\Windows\Fonts\arial.ttf"
        if os.path.exists(windows_font):
            shutil.copy(windows_font, FONT_PATH)
            print(f"[OK] Шрифт копиран от Windows -> {FONT_PATH}")
        else:
            print("[WARN] Шрифтът не е намерен, ще се използва default.")

def get_font(size: int):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()

def put_text_unicode(frame: np.ndarray, text: str, pos: Tuple[int,int],
                     font_size: int = 20, color: Tuple = (255,255,255),
                     bold: bool = False) -> None:
    # Премахване на емоджита – задържаме само ASCII + латиница + кирилица
    cleaned = ""
    for ch in text:
        cp = ord(ch)
        # Emoji диапазони – пропускаме ги
        if (0x1F300 <= cp <= 0x1FAFF) or (0x2600 <= cp <= 0x27BF) or (0xFE00 <= cp <= 0xFE0F):
            continue
        cleaned += ch
    text = cleaned

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    font    = get_font(font_size)
    draw.text((pos[0]+1, pos[1]+1), text, font=font, fill=(0, 0, 0, 180))
    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
    result  = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    np.copyto(frame, result)


# ──────────────────────────────────────────────
# Помощна: зареждане на изображение с contain (без разтягане)
# Поддържа PNG с алфа канал (за лого)
# ──────────────────────────────────────────────

def load_image_contain(path: str, target_w: int, target_h: int,
                        bg_color: Tuple = (0, 0, 0)) -> np.ndarray:
    """
    Зарежда изображение и го поставя в target_w x target_h canvas
    с contain логика (запазва aspect ratio, центрира, без разтягане).
    PNG с прозрачност се компостира върху bg_color.
    """
    # Опит за зареждане с алфа
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        # Fallback – черен canvas
        canvas = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
        put_text_unicode(canvas, f"[ЛИПСВА: {os.path.basename(path)}]",
                         (10, target_h // 2 - 10), font_size=14, color=(80, 80, 80))
        return canvas

    # Нормализация на каналите -> BGR + alpha маска
    if raw.ndim == 2:
        # Grayscale
        bgr   = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        alpha = None
    elif raw.shape[2] == 4:
        bgr   = raw[:, :, :3]
        alpha = raw[:, :, 3]
    else:
        bgr   = raw[:, :, :3]
        alpha = None

    src_h, src_w = bgr.shape[:2]

    # Portrait / landscape адаптация – contain scaling
    scale   = min(target_w / src_w, target_h / src_h)
    new_w   = max(1, int(src_w * scale))
    new_h   = max(1, int(src_h * scale))

    bgr_r = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if alpha is not None:
        alpha_r = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Canvas с фоново цвят
    canvas = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)

    # Центриране
    off_x = (target_w - new_w) // 2
    off_y = (target_h - new_h) // 2

    if alpha is not None:
        # Alpha compositing
        a = alpha_r.astype(np.float32) / 255.0
        for c in range(3):
            canvas[off_y:off_y+new_h, off_x:off_x+new_w, c] = (
                bgr_r[:, :, c].astype(np.float32) * a +
                bg_color[c] * (1.0 - a)
            ).astype(np.uint8)
    else:
        canvas[off_y:off_y+new_h, off_x:off_x+new_w] = bgr_r

    return canvas


# ──────────────────────────────────────────────
# Процедурно генериране на изображения (fallback)
# ──────────────────────────────────────────────

def _make_gradient_image(w: int, h: int,
                          c1: Tuple, c2: Tuple, c3: Tuple,
                          style: str = "diagonal") -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            if style == "diagonal":
                t = (x / w + y / h) / 2.0
            elif style == "radial":
                dx = (x - w/2) / (w/2)
                dy = (y - h/2) / (h/2)
                t  = min(1.0, math.sqrt(dx*dx + dy*dy))
            elif style == "vertical":
                t = y / h
            else:
                t = x / w

            if t < 0.5:
                s = t * 2
                col = tuple(c1[i] * (1-s) + c2[i] * s for i in range(3))
            else:
                s = (t - 0.5) * 2
                col = tuple(c2[i] * (1-s) + c3[i] * s for i in range(3))
            img[y, x] = col

    return np.clip(img, 0, 255).astype(np.uint8)

def _add_noise(img, amount=8.0):
    noise = np.random.randn(*img.shape) * amount
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def _add_circles(img, color, count=12):
    h, w = img.shape[:2]; out = img.copy()
    for _ in range(count):
        cx = random.randint(0, w); cy = random.randint(0, h)
        r  = random.randint(20, min(w, h) // 3)
        al = random.uniform(0.08, 0.22)
        ov = out.copy(); cv2.circle(ov, (cx, cy), r, color, -1)
        cv2.addWeighted(ov, al, out, 1-al, 0, out)
    return out

def _add_stars(img, count=60):
    h, w = img.shape[:2]; out = img.copy()
    for _ in range(count):
        x = random.randint(0, w-1); y = random.randint(0, h-1)
        r = random.randint(1, 3); br = random.randint(180, 255)
        cv2.circle(out, (x, y), r, (br, br, br), -1)
    return out

def _add_waves(img, color, count=5):
    h, w = img.shape[:2]; out = img.copy()
    for i in range(count):
        pts = []; amp = random.randint(15, 40)
        freq = random.uniform(0.01, 0.04); offset = random.randint(50, h-50)
        for x in range(0, w, 4):
            y2 = int(offset + amp * math.sin(freq * x + i))
            pts.append((x, max(0, min(h-1, y2))))
        for j in range(len(pts)-1):
            al = random.uniform(0.1, 0.25); ov = out.copy()
            cv2.line(ov, pts[j], pts[j+1], color, 2)
            cv2.addWeighted(ov, al, out, 1-al, 0, out)
    return out

def _add_triangles(img, color, count=8):
    h, w = img.shape[:2]; out = img.copy()
    for _ in range(count):
        pts = np.array([[random.randint(0,w), random.randint(0,h)],
                        [random.randint(0,w), random.randint(0,h)],
                        [random.randint(0,w), random.randint(0,h)]], np.int32)
        al = random.uniform(0.06, 0.18); ov = out.copy()
        cv2.fillPoly(ov, [pts], color)
        cv2.addWeighted(ov, al, out, 1-al, 0, out)
    return out


@dataclass
class ImageTheme:
    key:   str
    label: str
    image: Optional[np.ndarray] = field(default=None, repr=False)

    def build(self, w: int, h: int) -> None:
        self.image = _generate_theme_image(self.key, w, h)


def _generate_theme_image(key: str, w: int, h: int) -> np.ndarray:
    img_path = f"puzzle_images/{key}.jpg"
    png_path  = f"puzzle_images/{key}.png"

    for path in [img_path, png_path]:
        if os.path.exists(path):
            return load_image_contain(path, w, h, bg_color=(10, 10, 10))

    # Fallback – тъмен градиент ако файлът липсва
    return _make_gradient_image(w, h, (80,80,80), (160,160,180), (40,40,60))


IMAGE_THEMES: List[ImageTheme] = [
    ImageTheme("ocean",    "Океан"),
    ImageTheme("sunset",   "Залез"),
    ImageTheme("forest",   "Гора"),
    ImageTheme("space",    "Космос"),
    ImageTheme("candy",    "Бонбони"),
    ImageTheme("mountain", "Планини"),
    ImageTheme("abstract", "Абстракт"),
    ImageTheme("beach",    "Плаж"),
    ImageTheme("logo",     "Лого"),
]

# ──────────────────────────────────────────────
# Конфигурация
# ──────────────────────────────────────────────

pyautogui.FAILSAFE = True
pyautogui.PAUSE    = 0

SCREEN_W, SCREEN_H = pyautogui.size()

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

@dataclass
class Config:
    camera_index:   int   = 0
    cam_width:      int   = 1280
    cam_height:     int   = 720
    fps_limit:      int   = 30
    smooth_window:  int   = 6
    active_zone_x:  Tuple = (0.1, 0.9)
    active_zone_y:  Tuple = (0.1, 0.9)
    click_distance: float = 0.05
    click_cooldown: float = 0.4
    scroll_sens:    float = 400.0
    scroll_dead:    float = 0.008
    double_click_window: float = 0.40   # макс. сек между два пинча за двоен клик
    drag_hold_time: float = 0.55        # сек задържан пинч преди drag

CFG = Config()

# ──────────────────────────────────────────────
# MediaPipe индекси
# ──────────────────────────────────────────────

THUMB_TIP  = 4
INDEX_TIP  = 8;  INDEX_PIP  = 6
MIDDLE_TIP = 12; MIDDLE_PIP = 10
RING_TIP   = 16; RING_PIP   = 14
PINKY_TIP  = 20; PINKY_PIP  = 18

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

# ──────────────────────────────────────────────
# Помощни функции
# ──────────────────────────────────────────────

def download_model() -> None:
    if os.path.exists(MODEL_PATH):
        return
    print(f"Изтеглям модела... ({MODEL_URL})")
    r = requests.get(MODEL_URL, stream=True)
    r.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"[OK] Моделът е записан -> {MODEL_PATH}")


def lm_xy(landmarks, idx: int) -> Tuple[float, float]:
    lm = landmarks[idx]
    return lm.x, lm.y

def dist(p1, p2) -> float:
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def finger_up(landmarks, tip: int, pip: int) -> bool:
    return landmarks[tip].y < landmarks[pip].y

def map_to_screen(nx: float, ny: float) -> Tuple[int, int]:
    ax0, ax1 = CFG.active_zone_x
    ay0, ay1 = CFG.active_zone_y
    nx = max(ax0, min(nx, ax1))
    ny = max(ay0, min(ny, ay1))
    sx = int(np.interp(nx, [ax0, ax1], [0, SCREEN_W]))
    sy = int(np.interp(ny, [ay0, ay1], [0, SCREEN_H]))
    return sx, sy

def draw_rounded_rect(frame, x1, y1, x2, y2, color, radius=10, thickness=-1):
    if thickness == -1:
        cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(frame, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(frame, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(frame, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(frame, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.ellipse(frame, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(frame, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(frame, (x1 + radius, y2 - radius), (radius, radius),  90, 0, 90, color, thickness)
        cv2.ellipse(frame, (x2 - radius, y2 - radius), (radius, radius),   0, 0, 90, color, thickness)

# ──────────────────────────────────────────────
# Жест-детектор
# ──────────────────────────────────────────────

def detect_gesture(landmarks) -> str:
    index_up  = finger_up(landmarks, INDEX_TIP,  INDEX_PIP)
    middle_up = finger_up(landmarks, MIDDLE_TIP, MIDDLE_PIP)
    ring_up   = finger_up(landmarks, RING_TIP,   RING_PIP)
    pinky_up  = finger_up(landmarks, PINKY_TIP,  PINKY_PIP)

    thumb_pos = lm_xy(landmarks, THUMB_TIP)
    index_pos = lm_xy(landmarks, INDEX_TIP)
    pinch     = dist(thumb_pos, index_pos) < CFG.click_distance

    if pinch:
        return "click"
    if index_up and middle_up and ring_up and not pinky_up:
        return "right_click"
    if index_up and middle_up and not ring_up and not pinky_up:
        return "scroll"
    if index_up and not middle_up and not ring_up and not pinky_up:
        return "move"
    return "none"

# ──────────────────────────────────────────────
# Туториал режим (без емоджита в иконите)
# ──────────────────────────────────────────────

@dataclass
class TutorialStep:
    title:       str
    description: str
    gesture:     str
    icon_func:   object
    hold_time:   float = 1.5

TUTORIAL_BG   = (15, 15, 40)
TUTORIAL_ACNT = (0, 200, 255)
TUTORIAL_OK   = (0, 220, 100)
TUTORIAL_WARN = (0, 120, 255)

def _icon_move(frame, cx, cy):
    cv2.arrowedLine(frame, (cx, cy+30), (cx+40, cy-20), (0,220,120), 3, tipLength=0.35)
    cv2.circle(frame, (cx, cy+30), 8, (255,255,255), -1)
    # Замест емоджи – малък правоъгълник символизиращ пръст
    cv2.rectangle(frame, (cx-6, cy-50), (cx+6, cy-20), (255,220,80), -1)

def _icon_pinch(frame, cx, cy):
    cv2.circle(frame, (cx-15, cy), 9, (255,255,255), -1)
    cv2.circle(frame, (cx+15, cy), 9, (255,255,255), -1)
    cv2.line(frame,   (cx-6,  cy), (cx+6, cy), (0,180,255), 2)
    # Символ на щипка – две линии, сближаващи се
    cv2.line(frame, (cx-20, cy-30), (cx, cy-10), (0,180,255), 3)
    cv2.line(frame, (cx+20, cy-30), (cx, cy-10), (0,180,255), 3)

def _icon_two_fingers(frame, cx, cy):
    cv2.line(frame, (cx-12, cy+20), (cx-12, cy-25), (255,255,255), 5)
    cv2.line(frame, (cx+12, cy+20), (cx+12, cy-25), (255,255,255), 5)
    cv2.arrowedLine(frame, (cx+35, cy+10), (cx+35, cy-25), (255,220,0), 3, tipLength=0.4)
    cv2.arrowedLine(frame, (cx+45, cy-10), (cx+45, cy+25), (255,220,0), 3, tipLength=0.4)

def _icon_three_fingers(frame, cx, cy):
    for dx in (-16, 0, 16):
        cv2.line(frame, (cx+dx, cy+20), (cx+dx, cy-25), (255,255,255), 4)
    put_text_unicode(frame, "3", (cx-8, cy-48), font_size=26, color=(255,100,60))

def _icon_puzzle(frame, cx, cy):
    pts = np.array([
        [cx-20, cy-15], [cx, cy-15], [cx, cy-25],
        [cx+10, cy-25], [cx+10, cy-15], [cx+20, cy-15],
        [cx+20, cy+15], [cx, cy+15], [cx, cy+25],
        [cx-10, cy+25], [cx-10, cy+15], [cx-20, cy+15],
    ], np.int32)
    cv2.fillPoly(frame, [pts], (80, 140, 220))
    cv2.polylines(frame, [pts], True, (255,255,255), 2)


TUTORIAL_STEPS: List[TutorialStep] = [
    TutorialStep(
        title="Стъпка 1: Движение",
        description="Вдигнете само показалеца.\nКурсорът следва пръста.",
        gesture="move", icon_func=_icon_move, hold_time=2.0,
    ),
    TutorialStep(
        title="Стъпка 2: Ляв клик",
        description="Свийте палеца и показалеца заедно.\nПинч = ляв клик.",
        gesture="click", icon_func=_icon_pinch, hold_time=1.5,
    ),
    TutorialStep(
        title="Стъпка 3: Скрол",
        description="Вдигнете показалеца и средния.\nДвижете нагоре/надолу.",
        gesture="scroll", icon_func=_icon_two_fingers, hold_time=2.0,
    ),
    TutorialStep(
        title="Стъпка 4: Десен клик",
        description="Вдигнете три пръста.\nТова е десният клик.",
        gesture="right_click", icon_func=_icon_three_fingers, hold_time=1.5,
    ),
    TutorialStep(
        title="Стъпка 5: Пъзел режим",
        description="Натиснете P за пъзел.\nN = следваща тема.",
        gesture="click", icon_func=_icon_puzzle, hold_time=1.0,
    ),
]


class TutorialMode:
    def __init__(self):
        self.step_idx   = 0
        self.hold_start: Optional[float] = None
        self.all_done   = False

    @property
    def current(self) -> TutorialStep:
        return TUTORIAL_STEPS[self.step_idx]

    def update(self, gesture: str) -> bool:
        if self.all_done:
            return False
        step = self.current
        if gesture == step.gesture:
            if self.hold_start is None:
                self.hold_start = time.time()
            if time.time() - self.hold_start >= step.hold_time:
                self.hold_start = None
                self.step_idx  += 1
                if self.step_idx >= len(TUTORIAL_STEPS):
                    self.all_done = True
                return True
        else:
            self.hold_start = None
        return False

    def progress_ratio(self, gesture: str) -> float:
        if gesture != self.current.gesture or self.hold_start is None:
            return 0.0
        return min(1.0, (time.time() - self.hold_start) / self.current.hold_time)

    def draw(self, frame: np.ndarray, gesture: str, landmarks) -> None:
        h, w = frame.shape[:2]
        # Скалиране на панела спрямо размера на прозореца
        scale    = min(w / 640, h / 480)
        panel_w  = int(300 * scale)
        panel_h  = int(240 * scale)
        alpha    = 0.72
        overlay  = frame.copy()
        px, py   = 8, h - panel_h - 8
        draw_rounded_rect(overlay, px, py, px+panel_w, py+panel_h, TUTORIAL_BG, radius=14)
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

        fs_sm = max(11, int(13 * scale))
        fs_md = max(12, int(15 * scale))
        fs_lg = max(14, int(18 * scale))

        if self.all_done:
            put_text_unicode(frame, "Туториалът е завършен!", (px+14, py+int(90*scale)),
                             font_size=fs_md, color=(0,255,120))
            put_text_unicode(frame, "Натиснете T за изход", (px+14, py+int(122*scale)),
                             font_size=fs_sm, color=(180,180,180))
            return

        step = self.current
        put_text_unicode(frame, step.title, (px+12, py+10), font_size=fs_md, color=(0,200,255))
        for i, line in enumerate(step.description.split("\n")):
            put_text_unicode(frame, line, (px+12, py+34+i*int(20*scale)),
                             font_size=fs_sm, color=(210,210,210))

        bar_x = px + 12
        bar_y = py + panel_h - int(46 * scale)
        bar_w = panel_w - 24
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+12), (40,40,60), -1)
        fill = int(bar_w * self.progress_ratio(gesture))
        if fill > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x+fill, bar_y+12), TUTORIAL_OK, -1)
        draw_rounded_rect(frame, bar_x, bar_y, bar_x+bar_w, bar_y+12, (80,80,100), radius=4, thickness=1)

        if gesture == step.gesture:
            status = f"Браво! ({self.progress_ratio(gesture)*100:.0f}%)"
            color  = TUTORIAL_OK
        else:
            status = "Покажете жеста ->"
            color  = (180, 180, 0)
        put_text_unicode(frame, status, (px+12, py+panel_h-int(26*scale)),
                         font_size=fs_sm, color=color)

        total = len(TUTORIAL_STEPS)
        for i in range(total):
            cx2 = px + 12 + i * int(26 * scale)
            cy2 = py + panel_h - int(60 * scale)
            col = TUTORIAL_OK if i < self.step_idx else \
                  TUTORIAL_ACNT if i == self.step_idx else (60,60,80)
            cv2.circle(frame, (cx2, cy2), int(7*scale), col, -1)

        step.icon_func(frame, px + panel_w - int(50*scale), py + int(85*scale))


# ──────────────────────────────────────────────
# Избор на тема
# ──────────────────────────────────────────────

class ThemeSelector:
    COLS = 4
    PAD  = 8

    def __init__(self, frame_w: int, frame_h: int):
        self.fw = frame_w
        self.fh = frame_h
        # Размер на миниатюрите – скалира се спрямо прозореца
        self.THUMB_W = max(80, frame_w // 10)
        self.THUMB_H = max(56, frame_h // 9)
        self.thumbs: List[np.ndarray] = []
        self._build_thumbs(frame_w, frame_h)

    def _build_thumbs(self, fw: int, fh: int):
        bw = fw // 2
        bh = fh
        self.thumbs = []
        for theme in IMAGE_THEMES:
            if theme.image is None:
                theme.build(bw, bh)
            # Миниатюра с contain логика
            thumb = np.full((self.THUMB_H, self.THUMB_W, 3), (20, 20, 40), dtype=np.uint8)
            src   = theme.image
            sh, sw = src.shape[:2]
            sc     = min(self.THUMB_W / sw, self.THUMB_H / sh)
            nw, nh = max(1, int(sw*sc)), max(1, int(sh*sc))
            resized = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA)
            ox = (self.THUMB_W - nw) // 2
            oy = (self.THUMB_H - nh) // 2
            thumb[oy:oy+nh, ox:ox+nw] = resized
            self.thumbs.append(thumb)

    def draw(self, frame: np.ndarray, current_idx: int) -> None:
        h, w = frame.shape[:2]
        rows  = math.ceil(len(IMAGE_THEMES) / self.COLS)
        total_w = self.COLS * (self.THUMB_W + self.PAD) + self.PAD
        total_h = rows * (self.THUMB_H + self.PAD + 18) + self.PAD + 44

        start_x = w // 2 - total_w // 2
        start_y = h // 2 - total_h // 2

        overlay = frame.copy()
        draw_rounded_rect(overlay, start_x - 10, start_y - 10,
                          start_x + total_w + 10, start_y + total_h + 10,
                          (10, 10, 30), radius=14)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

        put_text_unicode(frame, "Избери тема  (N = следваща  B = предишна  Enter = OK)",
                         (start_x, start_y - 4), font_size=13, color=(180, 200, 255))

        for i, theme in enumerate(IMAGE_THEMES):
            row = i // self.COLS
            col = i % self.COLS
            x   = start_x + self.PAD + col * (self.THUMB_W + self.PAD)
            y   = start_y + 44 + row * (self.THUMB_H + self.PAD + 18)

            frame[y:y+self.THUMB_H, x:x+self.THUMB_W] = self.thumbs[i]

            if i == current_idx:
                cv2.rectangle(frame, (x-3, y-3),
                              (x+self.THUMB_W+3, y+self.THUMB_H+3),
                              (0, 220, 255), 3)
                put_text_unicode(frame, "v", (x + self.THUMB_W//2 - 4, y - 18),
                                 font_size=13, color=(0, 220, 255))
            else:
                cv2.rectangle(frame, (x, y), (x+self.THUMB_W, y+self.THUMB_H),
                              (80, 80, 100), 1)

            put_text_unicode(frame, theme.label,
                             (x + 2, y + self.THUMB_H + 2),
                             font_size=11, color=(200, 200, 220))


# ──────────────────────────────────────────────
# Пъзел
# ──────────────────────────────────────────────

@dataclass
class PuzzlePiece:
    id:       int
    img:      np.ndarray
    target_x: int
    target_y: int
    cur_x:    int
    cur_y:    int
    w:        int
    h:        int
    number:   int
    placed:   bool = False

    def contains(self, px: int, py: int) -> bool:
        return self.cur_x <= px <= self.cur_x + self.w and \
               self.cur_y <= py <= self.cur_y + self.h

    def snap_check(self, snap_dist: int = 30) -> bool:
        return (abs(self.cur_x - self.target_x) < snap_dist and
                abs(self.cur_y - self.target_y) < snap_dist)


PUZZLE_DIFFICULTY = {          # (cols, rows)
    'лесно':   (2, 2),
    'нормално': (3, 3),
    'трудно':  (4, 4),
}
DIFFICULTY_KEYS  = list(PUZZLE_DIFFICULTY.keys())
RECORDS_FILE     = 'puzzle_records.json'


def load_records() -> dict:
    try:
        with open(RECORDS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def save_record(theme_label: str, difficulty: str, seconds: int) -> bool:
    """Записва рекорд. Връща True ако е нов рекорд."""
    records = load_records()
    key = f"{theme_label}_{difficulty}"
    prev = records.get(key, None)
    if prev is None or seconds < prev:
        records[key] = seconds
        try:
            with open(RECORDS_FILE, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return True
    return False


class PuzzleGame:
    SNAP_DIST = 32

    def __init__(self, frame_w: int, frame_h: int, theme_idx: int = 0,
                 difficulty: str = 'нормално'):
        self.fw         = frame_w
        self.fh         = frame_h
        self.theme_idx  = theme_idx
        self.difficulty = difficulty
        cols, rows      = PUZZLE_DIFFICULTY.get(difficulty, (3, 3))
        self.COLS       = cols
        self.ROWS       = rows

        self.board_x = frame_w // 2
        self.board_y = 0
        self.board_w = frame_w // 2
        self.board_h = frame_h

        self.pieces:       List[PuzzlePiece] = []
        self.held:         Optional[int]     = None
        self.hold_off_x    = 0
        self.hold_off_y    = 0
        self.completed     = False
        self.start_time    = time.time()
        self.end_time:     Optional[float] = None
        self.snap_anim:    dict = {}
        self.is_new_record = False
        # Подсказка: piece_id -> time кога е активирана
        self._hint_piece:  Optional[int]   = None
        self._hint_start:  float           = 0.0
        self._hint_hover_start: float      = 0.0
        self._hint_hover_id:    Optional[int] = None

        self._build_puzzle()

    def _get_source(self) -> np.ndarray:
        theme = IMAGE_THEMES[self.theme_idx % len(IMAGE_THEMES)]
        # Винаги генерираме на точния размер на пъзел зоната
        if theme.image is None or \
           theme.image.shape[1] != self.board_w or \
           theme.image.shape[0] != self.board_h:
            theme.build(self.board_w, self.board_h)
        return theme.image

    def _build_puzzle(self):
        bw, bh = self.board_w, self.board_h
        pw = bw // self.COLS
        ph = bh // self.ROWS
        source = self._get_source()

        self.pieces = []
        for r in range(self.ROWS):
            for c in range(self.COLS):
                tx = self.board_x + c * pw
                ty = self.board_y + r * ph
                piece_img = source[r*ph:(r+1)*ph, c*pw:(c+1)*pw].copy()

                cv2.rectangle(piece_img, (0,0), (pw-1, ph-1), (255,255,255), 2)

                num = r * self.COLS + c + 1
                img_pil = Image.fromarray(cv2.cvtColor(piece_img, cv2.COLOR_BGR2RGB))
                draw    = ImageDraw.Draw(img_pil)
                font    = get_font(max(14, pw // 6))
                draw.text((pw//2 - 7, ph//2 - 14), str(num), font=font, fill=(0,0,0,160))
                draw.text((pw//2 - 8, ph//2 - 15), str(num), font=font, fill=(255,255,255,220))
                piece_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

                self.pieces.append(PuzzlePiece(
                    id=r * self.COLS + c,
                    img=piece_img,
                    target_x=tx, target_y=ty,
                    cur_x=0, cur_y=0,
                    w=pw, h=ph,
                    number=num,
                ))
        self._shuffle()

    def _shuffle(self):
        margin = 10
        pw = self.pieces[0].w
        ph = self.pieces[0].h
        positions = []
        cols = max(1, (self.board_x - margin) // (pw + margin))
        rows = max(1, self.fh // (ph + margin))
        for r in range(rows):
            for c in range(cols):
                positions.append((
                    margin + c * (pw + margin),
                    margin + r * (ph + margin),
                ))
        random.shuffle(positions)
        for i, piece in enumerate(self.pieces):
            if i < len(positions):
                piece.cur_x, piece.cur_y = positions[i]
            else:
                piece.cur_x = random.randint(margin, max(margin, self.board_x - pw - margin))
                piece.cur_y = random.randint(margin, max(margin, self.fh - ph - margin))

    def try_grab(self, px: int, py: int):
        if self.held is not None:
            return
        for piece in reversed(self.pieces):
            if not piece.placed and piece.contains(px, py):
                self.held = piece.id
                self.hold_off_x = px - piece.cur_x
                self.hold_off_y = py - piece.cur_y
                self.pieces.remove(piece)
                self.pieces.append(piece)
                return

    def move_held(self, px: int, py: int):
        if self.held is None:
            return
        piece = next((p for p in self.pieces if p.id == self.held), None)
        if piece:
            piece.cur_x = px - self.hold_off_x
            piece.cur_y = py - self.hold_off_y

    def update_hint_hover(self, px: int, py: int) -> None:
        """Следи курсора – 2 сек задържане над парче -> подсказка."""
        if self.held is not None or self.completed:
            self._hint_hover_id    = None
            self._hint_hover_start = 0.0
            return
        hovered = None
        for piece in reversed(self.pieces):
            if not piece.placed and piece.contains(px, py):
                hovered = piece.id
                break
        now = time.time()
        if hovered == self._hint_hover_id and hovered is not None:
            if now - self._hint_hover_start >= 2.0:
                self._hint_piece = hovered
                self._hint_start = now
        else:
            self._hint_hover_id    = hovered
            self._hint_hover_start = now
        # Изчезва след 1.5 сек
        if self._hint_piece is not None and now - self._hint_start > 1.5:
            self._hint_piece = None

    def release(self):
        if self.held is None:
            return
        piece = next((p for p in self.pieces if p.id == self.held), None)
        if piece and piece.snap_check(self.SNAP_DIST):
            piece.cur_x  = piece.target_x
            piece.cur_y  = piece.target_y
            piece.placed = True
            self.snap_anim[piece.id] = time.time()
        self.held = None
        if all(p.placed for p in self.pieces):
            self.completed     = True
            self.end_time      = time.time()
            duration           = int(self.end_time - self.start_time)
            theme              = IMAGE_THEMES[self.theme_idx % len(IMAGE_THEMES)]
            self.is_new_record = save_record(theme.label, self.difficulty, duration)

    def reset(self, theme_idx: Optional[int] = None, difficulty: Optional[str] = None):
        if theme_idx is not None:
            self.theme_idx  = theme_idx
        if difficulty is not None:
            self.difficulty = difficulty
            c, r = PUZZLE_DIFFICULTY.get(self.difficulty, (3, 3))
            self.COLS, self.ROWS = c, r
        self.completed     = False
        self.end_time      = None
        self.held          = None
        self.snap_anim     = {}
        self.is_new_record = False
        self._hint_piece   = None
        self._hint_hover_start = 0.0
        self._hint_hover_id    = None
        self.start_time    = time.time()
        self._build_puzzle()

    def _draw_dashed_rect(self, frame, x1, y1, x2, y2, color, thickness, dash=8):
        pts = [(x1,y1,x2,y1),(x2,y1,x2,y2),(x2,y2,x1,y2),(x1,y2,x1,y1)]
        for ax,ay,bx,by in pts:
            length = math.sqrt((bx-ax)**2+(by-ay)**2)
            steps  = max(1, int(length/dash))
            for i in range(0, steps, 2):
                sx = int(ax+(bx-ax)*i/steps);       sy = int(ay+(by-ay)*i/steps)
                ex = int(ax+(bx-ax)*min(i+1,steps)/steps)
                ey = int(ay+(by-ay)*min(i+1,steps)/steps)
                cv2.line(frame,(sx,sy),(ex,ey),color,thickness)

    def draw(self, frame: np.ndarray,
             cursor_xy: Optional[Tuple[int,int]], is_grabbing: bool) -> None:
        h, w = frame.shape[:2]
        now  = time.time()
        theme = IMAGE_THEMES[self.theme_idx % len(IMAGE_THEMES)]
        scale = min(w / 640, h / 480)

        # Фон на пъзел зоната
        grad = np.zeros((h, self.board_w, 3), dtype=np.uint8)
        for y in range(h):
            v = int(15 + 20 * (y / h))
            grad[y, :] = [v, v, v+10]
        frame[:, self.board_x:] = cv2.addWeighted(
            grad, 0.6, frame[:, self.board_x:], 0.4, 0)

        # Целеви позиции
        for piece in self.pieces:
            if not piece.placed:
                pulse = int(60 + 30 * math.sin(now * 3 + piece.id))
                self._draw_dashed_rect(frame,
                    piece.target_x, piece.target_y,
                    piece.target_x + piece.w, piece.target_y + piece.h,
                    (0, pulse, 0), 1)
                put_text_unicode(frame, str(piece.number),
                    (piece.target_x + piece.w//2 - 6, piece.target_y + piece.h//2 - 8),
                    font_size=14, color=(0, 100, 0))

        # Парчетата
        for piece in self.pieces:
            x, y   = piece.cur_x, piece.cur_y
            x2, y2 = x + piece.w, y + piece.h
            src_x1 = max(0, -x);   src_y1 = max(0, -y)
            dst_x1 = max(0, x);    dst_y1 = max(0, y)
            dst_x2 = min(w, x2);   dst_y2 = min(h, y2)
            src_x2 = src_x1 + (dst_x2 - dst_x1)
            src_y2 = src_y1 + (dst_y2 - dst_y1)

            if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
                continue

            if piece.id == self.held:
                shadow_off = 6
                sr = frame[min(h-1,dst_y1+shadow_off):min(h,dst_y2+shadow_off),
                           min(w-1,dst_x1+shadow_off):min(w,dst_x2+shadow_off)]
                if sr.size > 0:
                    frame[min(h-1,dst_y1+shadow_off):min(h,dst_y2+shadow_off),
                          min(w-1,dst_x1+shadow_off):min(w,dst_x2+shadow_off)] = \
                        (sr.astype(np.float32) * 0.4).astype(np.uint8)

            frame[dst_y1:dst_y2, dst_x1:dst_x2] = \
                piece.img[src_y1:src_y2, src_x1:src_x2]

            if piece.placed:
                snap_t = self.snap_anim.get(piece.id, 0)
                age    = now - snap_t
                border = (0, int(255*(1-age/0.6)), 60) if age < 0.6 else (0, 160, 60)
                thick  = 3 if age < 0.6 else 2
            elif piece.id == self.held:
                border = (0, 220, 255); thick = 3
            else:
                border = (160, 160, 180); thick = 1

            cv2.rectangle(frame, (dst_x1, dst_y1), (dst_x2, dst_y2), border, thick)

            if piece.id == self.held:
                roi = frame[dst_y1:dst_y2, dst_x1:dst_x2]
                frame[dst_y1:dst_y2, dst_x1:dst_x2] = np.clip(
                    roi.astype(np.int32) + 30, 0, 255).astype(np.uint8)

        # Подсказка – призрачна целева позиция
        if self._hint_piece is not None:
            hp = next((p for p in self.pieces if p.id == self._hint_piece), None)
            if hp and not hp.placed:
                ghost = hp.img.copy()
                ghost = (ghost.astype(np.float32) * 0.35).astype(np.uint8)
                tx1, ty1 = hp.target_x, hp.target_y
                tx2, ty2 = tx1 + hp.w, ty1 + hp.h
                dx1 = max(0, tx1); dy1 = max(0, ty1)
                dx2 = min(w, tx2); dy2 = min(h, ty2)
                sx1 = dx1 - tx1; sy1 = dy1 - ty1
                sx2 = sx1 + (dx2 - dx1); sy2 = sy1 + (dy2 - dy1)
                if dx2 > dx1 and dy2 > dy1:
                    roi = frame[dy1:dy2, dx1:dx2]
                    frame[dy1:dy2, dx1:dx2] = np.clip(
                        roi.astype(np.int32) + ghost[sy1:sy2, sx1:sx2].astype(np.int32),
                        0, 255).astype(np.uint8)
                    cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 255, 255), 2)
                    put_text_unicode(frame, "Подсказка",
                        (dx1 + 4, dy1 + 4), font_size=12, color=(0, 255, 255))

        # Разделителна линия
        cv2.line(frame, (self.board_x, 0), (self.board_x, h), (60, 100, 140), 2)

        # Прогрес панел – скалиран
        placed  = sum(1 for p in self.pieces if p.placed)
        total   = len(self.pieces)
        elapsed = int(now - self.start_time)
        panel_x = self.board_x + 4
        panel_w2 = min(240, self.board_w - 8)

        fs_sm = max(10, int(12 * scale))
        fs_md = max(12, int(14 * scale))

        draw_rounded_rect(frame, panel_x, 4, panel_x + panel_w2, 76, (20, 20, 35), radius=8)
        put_text_unicode(frame,
            f"{theme.label}: {placed}/{total}",
            (panel_x + 8, 8), font_size=fs_md, color=(0, 200, 255))
        put_text_unicode(frame, f"Време: {elapsed} сек  |  {self.difficulty}",
            (panel_x + 8, 30), font_size=fs_sm, color=(160, 200, 160))

        bar_x, bar_y = panel_x + 8, 56
        bar_w = panel_w2 - 16
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+10), (40,40,60), -1)
        fill_w = int(bar_w * placed / total) if total > 0 else 0
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x+fill_w, bar_y+10), (0,200,100), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+10), (80,80,100), 1)

        # Инструкции
        hints_y = h - 88
        draw_rounded_rect(frame, panel_x, hints_y, panel_x + panel_w2, h - 4, (20,20,35), radius=8)
        put_text_unicode(frame, "N = следваща тема",   (panel_x+8, hints_y+6),  font_size=fs_sm, color=(140,140,160))
        put_text_unicode(frame, "B = предишна тема",   (panel_x+8, hints_y+22), font_size=fs_sm, color=(140,140,160))
        put_text_unicode(frame, "R = разбъркай",       (panel_x+8, hints_y+38), font_size=fs_sm, color=(140,140,160))
        put_text_unicode(frame, "M = меню с теми",     (panel_x+8, hints_y+54), font_size=fs_sm, color=(140,140,160))
        put_text_unicode(frame, "P = изход от пъзел",  (panel_x+8, hints_y+70), font_size=fs_sm, color=(140,140,160))
        put_text_unicode(frame, "D = трудност",         (panel_x+8, hints_y+86), font_size=fs_sm, color=(140,140,160))

        # Курсор
        if cursor_xy:
            cx2, cy2 = cursor_xy
            pulse_c = int(200 + 55 * math.sin(now * 6))
            color   = (0, pulse_c, 255) if is_grabbing else (255, 255, 255)
            cv2.circle(frame, (cx2, cy2), 14, color, 2)
            cv2.circle(frame, (cx2, cy2),  3, color, -1)
            cv2.line(frame, (cx2-20, cy2), (cx2-8,  cy2), color, 1)
            cv2.line(frame, (cx2+8,  cy2), (cx2+20, cy2), color, 1)
            cv2.line(frame, (cx2, cy2-20), (cx2, cy2-8),  color, 1)
            cv2.line(frame, (cx2, cy2+8),  (cx2, cy2+20), color, 1)
            if is_grabbing:
                put_text_unicode(frame, "ХВАНАТО", (cx2+18, cy2-12),
                                 font_size=13, color=(0, 220, 255))

        # Победа
        if self.completed and self.end_time:
            self._draw_win_banner(frame, int(self.end_time - self.start_time))

    def _draw_win_banner(self, frame: np.ndarray, duration: int):
        h, w = frame.shape[:2]
        theme = IMAGE_THEMES[self.theme_idx % len(IMAGE_THEMES)]
        scale = min(w / 640, h / 480)
        overlay = frame.copy()
        bx1, by1 = w//4 - 20, h//3 - 10
        bx2, by2 = 3*w//4 + 20, 2*h//3 + 40
        draw_rounded_rect(overlay, bx1, by1, bx2, by2, (10, 50, 20), radius=16)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        draw_rounded_rect(frame, bx1, by1, bx2, by2, (0, 200, 80), radius=16, thickness=3)
        put_text_unicode(frame, "БРАВО!",
            (bx1 + int(60*scale), by1 + 20), font_size=int(28*scale), color=(0, 255, 120))
        put_text_unicode(frame, f"Завършен за {duration} секунди",
            (bx1 + int(30*scale), by1 + int(65*scale)), font_size=int(18*scale), color=(200, 240, 200))
        put_text_unicode(frame, f"Тема: {theme.label}  |  Трудност: {self.difficulty}",
            (bx1 + int(30*scale), by1 + int(95*scale)), font_size=int(14*scale), color=(160, 200, 160))
        # Рекорд
        key = f"{theme.label}_{self.difficulty}"
        rec = load_records().get(key)
        if rec:
            rec_color = (0, 255, 200) if self.is_new_record else (140, 180, 140)
            rec_text  = f"Рекорд: {rec} сек" + ("  <<< НОВ РЕКОРД!" if self.is_new_record else "")
            put_text_unicode(frame, rec_text,
                (bx1 + int(30*scale), by1 + int(120*scale)), font_size=int(13*scale), color=rec_color)
        put_text_unicode(frame, "N = нова тема  |  R = пак  |  D = трудност",
            (bx1 + int(20*scale), by1 + int(148*scale)), font_size=int(13*scale), color=(120, 160, 120))


# ──────────────────────────────────────────────
# Помощна: contain от numpy array
# ──────────────────────────────────────────────

def load_image_contain_from_array(src: np.ndarray, target_w: int, target_h: int,
                                   bg_color: Tuple = (15, 15, 25)) -> np.ndarray:
    """Поставя съществуващ BGR image в canvas с contain логика."""
    sh, sw = src.shape[:2]
    scale  = min(target_w / sw, target_h / sh)
    nw = max(1, int(sw * scale))
    nh = max(1, int(sh * scale))
    resized = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas  = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
    ox = (target_w - nw) // 2
    oy = (target_h - nh) // 2
    canvas[oy:oy+nh, ox:ox+nw] = resized
    return canvas


# ──────────────────────────────────────────────
# HUD
# ──────────────────────────────────────────────

GESTURE_LABEL = {
    "move":         "Движение",
    "click":        "Пинч клик",
    "double_click": "Двоен клик",
    "drag":         "Задържане / Drag",
    "right_click":  "Десен клик",
    "scroll":       "Скрол",
    "none":         "-",
}
GESTURE_COLOR = {
    "move":         (0,255,80),
    "click":        (0,120,255),
    "double_click": (0,200,255),
    "drag":         (255,160,0),
    "right_click":  (255,100,0),
    "scroll":      (255,220,0),
    "none":        (160,160,160),
}


def draw_hand(frame: np.ndarray, landmarks, w: int, h: int) -> None:
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 100), 2)
    for i, (x, y) in enumerate(pts):
        color = (0, 120, 255) if i in (INDEX_TIP, THUMB_TIP) else (220, 220, 220)
        r = 6 if i in (INDEX_TIP, THUMB_TIP) else 4
        cv2.circle(frame, (x, y), r, color, -1)


def draw_overlay(frame: np.ndarray, gesture: str, fps: float,
                 puzzle_mode: bool, tutorial_mode: bool,
                 presentation_mode: bool = False,
                 drawing_mode: bool = False) -> None:
    h, w = frame.shape[:2]
    scale = min(w / 640, h / 480)
    fs_sm = max(9,  int(11 * scale))
    fs_md = max(11, int(13 * scale))

    # Активна зона – само в нормален режим
    if not puzzle_mode and not tutorial_mode:
        ax0 = int(CFG.active_zone_x[0] * w)
        ax1 = int(CFG.active_zone_x[1] * w)
        ay0 = int(CFG.active_zone_y[0] * h)
        ay1 = int(CFG.active_zone_y[1] * h)
        cv2.rectangle(frame, (ax0, ay0), (ax1, ay1), (0, 200, 80), 1)

    # ── Компактна лента горе-ляво (само 1 ред) ──
    color = GESTURE_COLOR.get(gesture, (160, 160, 160))

    mode_str = ""
    if puzzle_mode:        mode_str = "[ПЪЗЕЛ] "
    elif tutorial_mode:    mode_str = "[УРОК] "
    elif presentation_mode:mode_str = "[ПРЕЗЕНТАЦИЯ] "
    elif drawing_mode:     mode_str = "[РИСУВАНЕ] "

    gesture_label = GESTURE_LABEL.get(gesture, '?')
    hud_text = f"{mode_str}{gesture_label}  |  FPS:{fps:.0f}  |  H=Начало  S=Настр  Q=Изход"
    bar_h = max(22, int(26 * scale))
    # Полупрозрачна лента само зад текста – минимална ширина
    text_w = min(len(hud_text) * int(7 * scale) + 16, w - 4)
    overlay = frame.copy()
    draw_rounded_rect(overlay, 4, 2, 4 + text_w, 2 + bar_h, (10, 10, 25), radius=6)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    put_text_unicode(frame, hud_text, (10, 4), font_size=fs_md, color=color)

    # ── Малка легенда горе-дясно – само когато не е в режим ──
    if not puzzle_mode and not tutorial_mode and not presentation_mode and not drawing_mode:
        hints = [
            "T=Урок  P=Пъзел",
            "I=Презент  W=Рисув",
            "Пинч=Клик  3пр=РКлик",
        ]
        hx = w - min(185, w // 3)
        panel_h2 = len(hints) * int(18 * scale) + 6
        overlay2 = frame.copy()
        draw_rounded_rect(overlay2, hx - 4, 2, w - 2, 2 + panel_h2, (10, 10, 25), radius=6)
        cv2.addWeighted(overlay2, 0.65, frame, 0.35, 0, frame)
        for i, hint in enumerate(hints):
            put_text_unicode(frame, hint, (hx, 6 + i * int(18 * scale)),
                             font_size=fs_sm, color=(130, 160, 130))


# ──────────────────────────────────────────────
# Режим Презентация
# ──────────────────────────────────────────────

class PresentationMode:
    """Жестово управление на слайдове.
    Пинч (click)         → Следващ слайд
    3 пръста (right_click) → Предишен слайд
    Свайп наляво/надясно → запасен fallback
    """
    GESTURE_COOLDOWN = 0.9   # сек между команди

    def __init__(self):
        self._last_cmd      = 0.0
        self._prev_ix       = None
        self._feedback_cmd  = None   # 'next' | 'prev'
        self._feedback_t    = 0.0    # кога е стартирал feedback
        self._feedback_dur  = 0.8    # сек показване на анимацията

    def update(self, gesture: str, ix: float) -> Optional[str]:
        """Детектира жест и връща 'next' | 'prev' | None."""
        now = time.time()
        if now - self._last_cmd < self.GESTURE_COOLDOWN:
            # Само обновяваме prev_ix за свайп fallback
            if gesture == "move":
                self._prev_ix = ix
            else:
                self._prev_ix = None
            return None

        cmd = None

        # Пинч → следващ
        if gesture == "click":
            cmd = "next"

        # 3 пръста → предишен
        elif gesture == "right_click":
            cmd = "prev"

        # Свайп fallback (само при "move")
        elif gesture == "move":
            if self._prev_ix is not None:
                delta = ix - self._prev_ix
                if delta > 0.14:
                    cmd = "prev"
                elif delta < -0.14:
                    cmd = "next"
            self._prev_ix = ix
        else:
            self._prev_ix = None

        if cmd:
            self._last_cmd     = now
            self._feedback_cmd = cmd
            self._feedback_t   = now
            if cmd == "next":
                pyautogui.press("right")
            else:
                pyautogui.press("left")

        return cmd

    def draw(self, frame: np.ndarray, last_cmd: Optional[str]) -> None:
        h, w = frame.shape[:2]
        scale = min(w / 640, h / 480)
        fs_sm = max(10, int(12 * scale))
        fs_md = max(13, int(16 * scale))

        # Компактен панел – само 2 реда
        panel_w = min(310, w - 12)
        draw_rounded_rect(frame, 6, 30, 6 + panel_w, 68, (30, 15, 40), radius=8)
        put_text_unicode(frame, "[ПРЕЗЕНТАЦИЯ]",
            (12, 34), font_size=fs_md, color=(200, 100, 255))
        put_text_unicode(frame, "Пинч=Следващ  3пръста=Предишен  Свайп=fallback",
            (12, 54), font_size=fs_sm, color=(180, 150, 200))

        # Визуална обратна връзка – голяма стрелка + цветен flash
        now = time.time()
        age = now - self._feedback_t
        if self._feedback_cmd and age < self._feedback_dur:
            alpha = max(0.0, 1.0 - age / self._feedback_dur)
            is_next = self._feedback_cmd == "next"
            arrow   = ">>" if is_next else "<<"
            color   = (0, 180, 255) if is_next else (255, 140, 0)

            # Flash overlay
            overlay = frame.copy()
            side_w  = w // 6
            if is_next:
                cv2.rectangle(overlay, (w - side_w, 0), (w, h), color, -1)
            else:
                cv2.rectangle(overlay, (0, 0), (side_w, h), color, -1)
            cv2.addWeighted(overlay, alpha * 0.25, frame, 1 - alpha * 0.25, 0, frame)

            # Голям текст в центъра
            fs_big = max(32, int(52 * scale))
            cx_off = -40 if is_next else 10
            put_text_unicode(frame, arrow,
                (w // 2 + cx_off, h // 2 - int(30 * scale)),
                font_size=fs_big, color=color)
            label = "СЛЕДВАЩ" if is_next else "ПРЕДИШЕН"
            put_text_unicode(frame, label,
                (w // 2 - int(45 * scale), h // 2 + int(30 * scale)),
                font_size=max(14, int(18 * scale)), color=color)



# ──────────────────────────────────────────────
# Режим Рисуване
# ──────────────────────────────────────────────

DRAW_COLORS = [
    (0,   255, 80),   # зелено
    (0,   120, 255),  # синьо
    (0,   220, 255),  # циан
    (255, 80,  0),    # оранжево
    (255, 255, 255),  # бяло
    (255, 60,  200),  # розово
    (255, 220, 0),    # жълто
]
DRAW_THICKNESS = [2, 4, 7, 12]


class DrawingMode:
    """Рисуване с показалеца; пинч = пауза; 2 пръста = смяна цвят."""
    def __init__(self):
        self.canvas:       Optional[np.ndarray] = None
        self.color_idx    = 0
        self.thick_idx    = 1
        self._prev_pt:    Optional[Tuple[int,int]] = None
        self._drawing     = False
        self._last_color_change = 0.0

    def _ensure_canvas(self, h: int, w: int):
        if self.canvas is None or self.canvas.shape[:2] != (h, w):
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)

    def update(self, gesture: str, ix: float, iy: float,
               frame_w: int, frame_h: int) -> None:
        self._ensure_canvas(frame_h, frame_w)
        px = int(ix * frame_w)
        py = int(iy * frame_h)
        now = time.time()

        if gesture == "scroll" and now - self._last_color_change > 0.5:
            self.color_idx = (self.color_idx + 1) % len(DRAW_COLORS)
            self._last_color_change = now
            self._prev_pt = None
            return

        if gesture == "move":
            self._drawing = True
            if self._prev_pt:
                cv2.line(self.canvas, self._prev_pt, (px, py),
                         DRAW_COLORS[self.color_idx],
                         DRAW_THICKNESS[self.thick_idx])
            self._prev_pt = (px, py)
        else:
            self._drawing = False
            self._prev_pt = None

    def clear(self):
        if self.canvas is not None:
            self.canvas[:] = 0

    def change_thickness(self):
        self.thick_idx = (self.thick_idx + 1) % len(DRAW_THICKNESS)

    def draw(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        self._ensure_canvas(h, w)
        # Налага canvas върху frame
        mask = self.canvas.any(axis=2)
        frame[mask] = cv2.addWeighted(frame, 0.3, self.canvas, 0.7, 0)[mask]

        # HUD – позициониран под главната лента
        scale = min(w / 640, h / 480)
        fs_sm = max(10, int(11 * scale))
        fs_md = max(12, int(14 * scale))
        panel_w = min(280, w - 12)
        draw_rounded_rect(frame, 6, 34, 6 + panel_w, 120, (20, 30, 15), radius=8)
        put_text_unicode(frame, "[РИСУВАНЕ]",
            (12, 38), font_size=fs_md, color=(80, 255, 120))
        color = DRAW_COLORS[self.color_idx]
        put_text_unicode(frame, f"Цвят (2 пръста = смяна)",
            (12, 60), font_size=fs_sm, color=color)
        thick = DRAW_THICKNESS[self.thick_idx]
        put_text_unicode(frame, f"Дебелина: {thick}px  (Z = смяна)",
            (12, 78), font_size=fs_sm, color=(180, 200, 180))
        put_text_unicode(frame, "C = изчисти  W = изход",
            (12, 96), font_size=fs_sm, color=(140, 160, 140))
        # Цветна точка
        cv2.circle(frame, (int(panel_w - 20), 79), 10, color, -1)
        cv2.circle(frame, (int(panel_w - 20), 79), 10, (255,255,255), 1)


# ──────────────────────────────────────────────
# Начална страница (Home Screen)
# ──────────────────────────────────────────────

HOME_MENU_ITEMS = [
    # (клавиш, label, описание, цвят)
    ("T", "Урок режим",       "Научи жестовете стъпка по стъпка",       (0,  200, 255)),
    ("P", "Пъзел режим",      "Наредете картинки с жестове",             (80, 180, 255)),
    ("I", "Презентация",      "Контролирайте слайдове с пинч/жест",      (200, 100, 255)),
    ("W", "Рисуване",         "Рисувайте с пръст във въздуха",           (80,  255, 120)),
    ("S", "Настройки",        "Чувствителност, скорост, плавност",       (255, 200, 60)),
    ("F", "Fullscreen",       "Превключи цял екран",                     (160, 160, 220)),
    ("Q", "Изход",            "Затвори програмата",                      (100, 100, 120)),
]

# Жестови иконки за home screen (рисуват се вдясно от реда)
def _draw_home_gesture_icon(frame, cx, cy, key):
    if key == "T":
        # Книжка / урок
        cv2.rectangle(frame, (cx-18, cy-16), (cx+18, cy+16), (0,180,220), 2)
        for dy in (-6, 0, 6):
            cv2.line(frame, (cx-12, cy+dy), (cx+12, cy+dy), (0,180,220), 1)
    elif key == "P":
        # Пъзел парче
        pts = np.array([[cx-14,cy-10],[cx,cy-10],[cx,cy-18],[cx+8,cy-18],
                        [cx+8,cy-10],[cx+14,cy-10],[cx+14,cy+10],[cx,cy+10],
                        [cx,cy+18],[cx-8,cy+18],[cx-8,cy+10],[cx-14,cy+10]], np.int32)
        cv2.polylines(frame, [pts], True, (80,160,255), 2)
    elif key == "I":
        # Слайд + стрелка
        cv2.rectangle(frame, (cx-16, cy-10), (cx+16, cy+10), (180,80,255), 2)
        cv2.arrowedLine(frame, (cx-10,cy), (cx+10,cy), (180,80,255), 2, tipLength=0.4)
    elif key == "W":
        # Вълниста линия (четка)
        pts2 = [(cx-16+i*4, cy + int(8*math.sin(i*1.2))) for i in range(9)]
        for i in range(len(pts2)-1):
            cv2.line(frame, pts2[i], pts2[i+1], (60,230,100), 2)
    elif key == "S":
        # Зъбно колело (апроксимация)
        cv2.circle(frame, (cx, cy), 12, (220,180,50), 2)
        cv2.circle(frame, (cx, cy),  5, (220,180,50), -1)
        for ang in range(0, 360, 45):
            r = math.radians(ang)
            x1 = int(cx + 12*math.cos(r)); y1 = int(cy + 12*math.sin(r))
            x2 = int(cx + 17*math.cos(r)); y2 = int(cy + 17*math.sin(r))
            cv2.line(frame, (x1,y1), (x2,y2), (220,180,50), 3)
    elif key == "F":
        # Четири ъгъла
        for sx, sy in [(-1,-1),(-1,1),(1,-1),(1,1)]:
            ox, oy = cx + sx*14, cy + sy*10
            cv2.line(frame, (ox, oy), (ox-sx*8, oy),   (140,140,200), 2)
            cv2.line(frame, (ox, oy), (ox, oy-sy*6),   (140,140,200), 2)
    elif key == "Q":
        # X
        cv2.line(frame, (cx-12,cy-10),(cx+12,cy+10),(100,100,130),2)
        cv2.line(frame, (cx+12,cy-10),(cx-12,cy+10),(100,100,130),2)


class HomeScreen:
    """Начална страница – показва всички режими с клавишни shortcuts."""

    ANIM_SPEED = 2.5   # скорост на пулсиране

    def __init__(self):
        self.visible    = True   # при старт е видим
        self._start_t   = time.time()
        self._selected  = 0     # текущо избран ред (за жест-навигация)
        self._hover_t   = [0.0] * len(HOME_MENU_ITEMS)  # hover timestamp за всеки ред

    def toggle(self):
        self.visible = not self.visible
        if self.visible:
            self._start_t = time.time()

    def draw(self, frame: np.ndarray, fps: float) -> None:
        if not self.visible:
            return
        h, w  = frame.shape[:2]
        scale = min(w / 640, h / 480)

        # Тъмен полупрозрачен фон над целия кадър
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (8, 12, 28), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

        now   = time.time()
        pulse = 0.5 + 0.5 * math.sin((now - self._start_t) * self.ANIM_SPEED)

        # ── Заглавие ──
        title_fs = max(18, int(26 * scale))
        sub_fs   = max(11, int(13 * scale))
        item_fs  = max(12, int(15 * scale))
        desc_fs  = max(10, int(11 * scale))
        key_fs   = max(14, int(17 * scale))

        title_color = (
            int(0   + 40  * pulse),
            int(180 + 40  * pulse),
            int(220 + 35  * pulse),
        )
        put_text_unicode(frame, "Camera Mouse Control",
            (w//2 - int(130*scale), int(18*scale)), font_size=title_fs, color=title_color)
        put_text_unicode(frame, "Управление с жестове на ръката  |  MediaPipe",
            (w//2 - int(145*scale), int(18*scale) + title_fs + 4),
            font_size=sub_fs, color=(100, 120, 160))

        # Малък разделител
        sep_y = int(18*scale) + title_fs + sub_fs + 14
        cv2.line(frame, (w//4, sep_y), (3*w//4, sep_y), (40, 60, 100), 1)

        # ── Менюта ──
        n      = len(HOME_MENU_ITEMS)
        row_h  = max(44, int(52 * scale))
        total_h= n * row_h
        start_y= sep_y + 10
        panel_w= min(580, w - 40)
        panel_x= w//2 - panel_w//2

        for i, (key, label, desc, color) in enumerate(HOME_MENU_ITEMS):
            ry  = start_y + i * row_h
            is_selected = (i == self._selected)

            # Фон на реда
            bg_alpha = 0.55 if is_selected else 0.30
            row_color = (int(color[0]*0.25), int(color[1]*0.25), int(color[2]*0.25))
            ov2 = frame.copy()
            draw_rounded_rect(ov2, panel_x, ry, panel_x + panel_w, ry + row_h - 4,
                              row_color, radius=10)
            cv2.addWeighted(ov2, bg_alpha, frame, 1 - bg_alpha, 0, frame)

            # Бордер за избрания ред
            if is_selected:
                pulse_col = tuple(min(255, int(c * (0.7 + 0.3*pulse))) for c in color)
                draw_rounded_rect(frame, panel_x, ry, panel_x + panel_w, ry + row_h - 4,
                                  pulse_col, radius=10, thickness=2)

            # Клавиш бокс
            kx = panel_x + 10
            ky = ry + (row_h - 4)//2 - int(14*scale)
            draw_rounded_rect(frame, kx, ky, kx + int(28*scale), ky + int(28*scale),
                              (30, 40, 70), radius=6)
            draw_rounded_rect(frame, kx, ky, kx + int(28*scale), ky + int(28*scale),
                              color, radius=6, thickness=1)
            put_text_unicode(frame, key,
                (kx + int(7*scale), ky + int(5*scale)), font_size=key_fs, color=color)

            # Label + описание
            tx = kx + int(36*scale)
            put_text_unicode(frame, label,
                (tx, ry + int(8*scale)), font_size=item_fs, color=color)
            put_text_unicode(frame, desc,
                (tx, ry + int(8*scale) + item_fs + 2), font_size=desc_fs,
                color=(140, 150, 170))

            # Икона вдясно
            icon_cx = panel_x + panel_w - int(32*scale)
            icon_cy = ry + (row_h - 4)//2
            _draw_home_gesture_icon(frame, icon_cx, icon_cy, key)

        # ── Долна лента ──
        foot_y = start_y + total_h + 8
        put_text_unicode(frame,
            "HOME  -  натиснете клавиш за да влезете в режим   |   ESC / H = затвори",
            (panel_x, foot_y), font_size=desc_fs, color=(70, 80, 110))

        # FPS малко горе-дясно
        put_text_unicode(frame, f"FPS {fps:.0f}",
            (w - int(60*scale), 6), font_size=desc_fs, color=(60, 70, 100))


# ──────────────────────────────────────────────
# Настройки (Settings) панел
# ──────────────────────────────────────────────

@dataclass
class SettingItem:
    label:  str
    attr:   str        # атрибут в CFG
    step:   float
    min_v:  float
    max_v:  float
    fmt:    str = "{:.2f}"


SETTINGS_ITEMS = [
    SettingItem("Плавност (smooth)", "smooth_window", 1, 2, 20, "{:.0f}"),
    SettingItem("Разст. клик",       "click_distance", 0.005, 0.02, 0.15),
    SettingItem("Cooldown клик",     "click_cooldown",  0.05, 0.1, 1.5),
    SettingItem("Чувств. скрол",     "scroll_sens",    20.0, 50.0, 1000.0, "{:.0f}"),
    SettingItem("Drag задържане",    "drag_hold_time",  0.05, 0.2, 2.0),
]


class SettingsPanel:
    def __init__(self):
        self.selected = 0
        self.visible  = False

    def toggle(self):
        self.visible = not self.visible

    def move(self, delta: int):
        self.selected = (self.selected + delta) % len(SETTINGS_ITEMS)

    def adjust(self, delta: int):
        item = SETTINGS_ITEMS[self.selected]
        val  = getattr(CFG, item.attr)
        val  = round(val + item.step * delta, 6)
        val  = max(item.min_v, min(item.max_v, val))
        setattr(CFG, item.attr, val)
        # Специален случай – smooth_window трябва да е int
        if item.attr == "smooth_window":
            setattr(CFG, item.attr, int(val))

    def draw(self, frame: np.ndarray) -> None:
        if not self.visible:
            return
        h, w  = frame.shape[:2]
        scale = min(w / 640, h / 480)
        fs_sm = max(10, int(12 * scale))
        fs_md = max(12, int(15 * scale))

        panel_w = min(360, w - 20)
        panel_h = 50 + len(SETTINGS_ITEMS) * 28 + 30
        sx = w // 2 - panel_w // 2
        sy = h // 2 - panel_h // 2

        overlay = frame.copy()
        draw_rounded_rect(overlay, sx, sy, sx + panel_w, sy + panel_h, (15, 20, 40), radius=12)
        cv2.addWeighted(overlay, 0.90, frame, 0.10, 0, frame)
        draw_rounded_rect(frame, sx, sy, sx + panel_w, sy + panel_h,
                          (80, 100, 160), radius=12, thickness=2)

        put_text_unicode(frame, "НАСТРОЙКИ  (S = затвори)",
            (sx + 12, sy + 10), font_size=fs_md, color=(160, 200, 255))
        put_text_unicode(frame, "Стрелки UP/DOWN = избор   LEFT/RIGHT = промяна",
            (sx + 12, sy + 32), font_size=max(9, int(10 * scale)), color=(120, 140, 180))

        for i, item in enumerate(SETTINGS_ITEMS):
            y  = sy + 56 + i * 28
            bg = (40, 50, 80) if i == self.selected else (20, 25, 45)
            draw_rounded_rect(frame, sx + 6, y - 2, sx + panel_w - 6, y + 22, bg, radius=6)
            val  = getattr(CFG, item.attr)
            vstr = item.fmt.format(val)
            col  = (0, 220, 255) if i == self.selected else (180, 190, 210)
            put_text_unicode(frame, f"{item.label}: {vstr}",
                (sx + 14, y + 2), font_size=fs_sm, color=col)
            if i == self.selected:
                put_text_unicode(frame, "< >",
                    (sx + panel_w - 40, y + 2), font_size=fs_sm, color=(255, 200, 0))

        put_text_unicode(frame, "ESC = затвори без промяна",
            (sx + 12, sy + panel_h - 20), font_size=max(9, int(10 * scale)),
            color=(100, 110, 130))


# ──────────────────────────────────────────────
# Главен контролер
# ──────────────────────────────────────────────

# Глобален прозорец – пълен екран
WINDOW_NAME = "Camera Mouse Control"

def setup_fullscreen_window():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


class CameraMouseController:
    def __init__(self):
        download_model()
        ensure_font()

        # Темите се генерират при нужда на правилния размер (не предварително на малък)

        self.pos_buffer      = deque(maxlen=CFG.smooth_window)
        self.last_click      = 0.0
        self.last_click2     = 0.0   # предпоследен клик (за двоен клик)
        self._drag_active    = False  # мишката е задържана
        self._pinch_held     = False  # пинчът е задържан в момента
        self._pinch_start_t  = 0.0   # кога е захванат пинчът
        self.scroll_ref_y:   Optional[float] = None
        self.scroll_accum:   float = 0.0
        self.latest_result:  Optional[HandLandmarkerResult] = None
        self.fps             = 0.0
        self._frame_count    = 0
        self._fps_timer      = time.time()

        self.puzzle_mode        = False
        self.tutorial_mode      = False
        self.presentation_mode  = False
        self.drawing_mode       = False
        self.show_theme_menu    = False
        self.puzzle:            Optional[PuzzleGame]      = None
        self.tutorial:          Optional[TutorialMode]    = None
        self.selector:          Optional[ThemeSelector]   = None
        self.presentation:      Optional[PresentationMode]= None
        self.drawing:           Optional[DrawingMode]     = None
        self.settings_panel:    SettingsPanel              = SettingsPanel()
        self.current_theme_idx  = 0
        self.puzzle_difficulty  = 'нормално'
        self._pres_last_cmd:    Optional[str] = None
        self._pres_last_t:      float = 0.0

        # Модален прозорец за обучение
        self._tut_prompt_mode:  Optional[str] = None  # 'tutorial'|'puzzle'|'presentation'|'drawing'
        self._tut_prompt_active: bool = False

        # Начална страница
        self.home_screen: HomeScreen = HomeScreen()  # видим при старт

        self.PINCH_CONFIRM = 3
        self.pinch_raw_buf = deque(maxlen=self.PINCH_CONFIRM)
        self.pinch_stable  = False
        self.pinch_prev    = False
        self.puzzle_cursor_buf = deque(maxlen=8)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

        def _callback(result: HandLandmarkerResult, _, timestamp_ms: int):
            self.latest_result = result

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionTaskRunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
            result_callback=_callback,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    def smooth(self, x: int, y: int) -> Tuple[int, int]:
        self.pos_buffer.append((x, y))
        return (
            int(np.mean([p[0] for p in self.pos_buffer])),
            int(np.mean([p[1] for p in self.pos_buffer])),
        )

    def do_move(self, nx: float, ny: float) -> None:
        sx, sy = map_to_screen(nx, ny)
        sx, sy = self.smooth(sx, sy)
        pyautogui.moveTo(sx, sy)

    def do_click(self, button: str = "left") -> None:
        """Единичен клик с cooldown. При бърз двоен пинч → double click."""
        now = time.time()
        if now - self.last_click < CFG.click_cooldown:
            return
        # Проверка за двоен клик
        if button == "left" and (now - self.last_click2) < CFG.double_click_window:
            pyautogui.doubleClick()
            self.last_click2 = 0.0  # нулираме за следващия цикъл
        else:
            pyautogui.click(button=button)
            self.last_click2 = self.last_click
        self.last_click = now

    def update_drag(self, pinch_now: bool, nx: float, ny: float) -> str:
        """
        Следи задържан пинч -> drag.
        Връща: 'drag' | 'click' | 'none' (жеста за HUD).
        """
        now = time.time()
        if pinch_now:
            if not self._pinch_held:
                # Нов пинч
                self._pinch_held    = True
                self._pinch_start_t = now
            held_dur = now - self._pinch_start_t
            if held_dur >= CFG.drag_hold_time:
                # Задържан достатъчно дълго -> drag
                if not self._drag_active:
                    self._drag_active = True
                    pyautogui.mouseDown(button='left')
                sx, sy = map_to_screen(nx, ny)
                sx, sy = self.smooth(sx, sy)
                pyautogui.moveTo(sx, sy)
                return 'drag'
            else:
                # Все още не е drag – само движи
                self.do_move(nx, ny)
                return 'click'  # ще стане клик при пускане
        else:
            if self._pinch_held:
                # Пинчът е пуснат
                if self._drag_active:
                    pyautogui.mouseUp(button='left')
                    self._drag_active = False
                else:
                    # Кратък пинч -> клик
                    self.do_click('left')
            self._pinch_held = False
            return 'none'

    def do_scroll(self, ny: float) -> None:
        if self.scroll_ref_y is None:
            self.scroll_ref_y = ny
            self.scroll_accum = 0.0
            return
        delta_norm = self.scroll_ref_y - ny
        self.scroll_accum += delta_norm * SCREEN_H
        scroll_units = int(self.scroll_accum / CFG.scroll_sens)
        if scroll_units != 0:
            pyautogui.scroll(scroll_units)
            self.scroll_accum -= scroll_units * CFG.scroll_sens
        self.scroll_ref_y = ny

    def reset_scroll(self):
        self.scroll_ref_y = None
        self.scroll_accum = 0.0

    def _smooth_puzzle_cursor(self, px: int, py: int) -> Tuple[int, int]:
        self.puzzle_cursor_buf.append((px, py))
        return (
            int(np.mean([p[0] for p in self.puzzle_cursor_buf])),
            int(np.mean([p[1] for p in self.puzzle_cursor_buf])),
        )

    def _update_pinch_debounce(self, raw: bool) -> bool:
        self.pinch_raw_buf.append(raw)
        if len(self.pinch_raw_buf) < self.PINCH_CONFIRM:
            return self.pinch_stable
        if all(self.pinch_raw_buf):
            self.pinch_stable = True
        elif not any(self.pinch_raw_buf):
            self.pinch_stable = False
        return self.pinch_stable

    def handle_puzzle(self, landmarks, frame_w: int, frame_h: int):
        ix, iy  = lm_xy(landmarks, INDEX_TIP)
        raw_px  = int(ix * frame_w)
        raw_py  = int(iy * frame_h)
        px, py  = self._smooth_puzzle_cursor(raw_px, raw_py)

        thumb_pos = lm_xy(landmarks, THUMB_TIP)
        index_pos = lm_xy(landmarks, INDEX_TIP)
        raw_pinch = dist(thumb_pos, index_pos) < CFG.click_distance
        pinch_now = self._update_pinch_debounce(raw_pinch)

        if pinch_now and not self.pinch_prev:
            self.puzzle.try_grab(px, py)
        elif not pinch_now and self.pinch_prev:
            self.puzzle.release()
        if pinch_now:
            self.puzzle.move_held(px, py)

        self.pinch_prev = pinch_now
        return px, py, pinch_now

    def _reset_pinch(self):
        self.pinch_raw_buf.clear()
        self.pinch_stable = False
        self.pinch_prev   = False
        self.puzzle_cursor_buf.clear()

    def update_fps(self) -> None:
        self._frame_count += 1
        elapsed = time.time() - self._fps_timer
        if elapsed >= 1.0:
            self.fps          = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_timer   = time.time()

    def _switch_theme(self, delta: int):
        self.current_theme_idx = (self.current_theme_idx + delta) % len(IMAGE_THEMES)
        if self.puzzle:
            self.puzzle.reset(theme_idx=self.current_theme_idx)
            self._reset_pinch()
        theme = IMAGE_THEMES[self.current_theme_idx]
        print(f"[Тема] {theme.label}")

    def _draw_tutorial_prompt(self, frame: np.ndarray) -> None:
        """Рисува модален прозорец 'Искаш ли обучение? (Y/N)'."""
        if not self._tut_prompt_active:
            return
        h, w  = frame.shape[:2]
        scale = min(w / 640, h / 480)
        pw, ph = min(420, w - 40), 120
        px = w // 2 - pw // 2
        py = h // 2 - ph // 2

        mode_labels = {
            'tutorial':     'Урок',
            'puzzle':       'Пъзел',
            'presentation': 'Презентация',
            'drawing':      'Рисуване',
        }
        label = mode_labels.get(self._tut_prompt_mode, 'Режим')

        overlay = frame.copy()
        draw_rounded_rect(overlay, px, py, px + pw, py + ph, (15, 30, 55), radius=14)
        cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)
        draw_rounded_rect(frame, px, py, px + pw, py + ph, (80, 130, 220), radius=14, thickness=2)

        fs_md = max(13, int(16 * scale))
        fs_sm = max(11, int(13 * scale))
        put_text_unicode(frame, f"Режим: {label}",
            (px + 16, py + 12), font_size=fs_md, color=(0, 200, 255))
        put_text_unicode(frame, "Искаш ли да видиш обучението?",
            (px + 16, py + 38), font_size=fs_sm, color=(220, 220, 220))
        put_text_unicode(frame, "Y = Да, покажи урока",
            (px + 16, py + 62), font_size=fs_sm, color=(0, 220, 100))
        put_text_unicode(frame, "N = Не, влез директно в режима",
            (px + 16, py + 82), font_size=fs_sm, color=(255, 160, 60))
        put_text_unicode(frame, "ESC = Отказ",
            (px + 16, py + 102), font_size=max(9, int(11 * scale)), color=(140, 140, 160))

    def _handle_tutorial_prompt_key(self, key: int, key32: int, w: int, h: int) -> bool:
        """Обработва клавиш докато модалът е активен. Връща True ако е 'изял' клавиша."""
        if not self._tut_prompt_active:
            return False
        if key == ord('y') or key == ord('Y'):
            self._tut_prompt_active = False
            self.home_screen.visible = False
            # Стартираме урока независимо от режима
            self.tutorial_mode = True
            self.tutorial = TutorialMode()
            self.puzzle_mode = False; self.drawing_mode = False
            self.presentation_mode = False; self.show_theme_menu = False
            print(f"[Урок] Стартиран преди '{self._tut_prompt_mode}'")
        elif key == ord('n') or key == ord('N'):
            self._tut_prompt_active = False
            self._enter_mode_direct(self._tut_prompt_mode, w, h)
        elif key == 27:  # ESC
            self._tut_prompt_active = False
            print("[Режим] Отказано.")
        return True

    def _enter_mode_direct(self, mode: str, w: int, h: int) -> None:
        """Влиза директно в режима без обучение."""
        self.tutorial_mode = False; self.puzzle_mode = False
        self.drawing_mode = False; self.presentation_mode = False
        self.show_theme_menu = False
        self.home_screen.visible = False
        if mode == 'tutorial':
            self.tutorial_mode = True
            self.tutorial = TutorialMode()
            print("[Режим] Урок")
        elif mode == 'puzzle':
            self.puzzle_mode = True
            self._reset_pinch()
            if self.puzzle is None:
                self.puzzle = PuzzleGame(w, h, theme_idx=self.current_theme_idx)
            elif self.puzzle.completed:
                self.puzzle.reset(theme_idx=self.current_theme_idx)
            print(f"[Режим] Пъзел - {IMAGE_THEMES[self.current_theme_idx].label}")
        elif mode == 'presentation':
            self.presentation_mode = True
            self.presentation = PresentationMode()
            print("[Режим] Презентация")
        elif mode == 'drawing':
            self.drawing_mode = True
            if self.drawing is None:
                self.drawing = DrawingMode()
            print("[Режим] Рисуване")

    def run(self) -> None:
        cap = cv2.VideoCapture(CFG.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CFG.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.cam_height)
        cap.set(cv2.CAP_PROP_FPS,          CFG.fps_limit)

        if not cap.isOpened():
            print("[ГРЕШКА] Камерата не може да бъде отворена!")
            return

        print(f"[OK] Камерата стартирана. Екран: {SCREEN_W}x{SCREEN_H}")
        print("     T=Урок | P=Пъзел | N=Следваща тема | B=Предишна тема")
        print("     M=Меню с теми | R=Разбъркай | Q=Изход\n")

        frame_delay  = 1.0 / CFG.fps_limit
        timestamp_ms = 0

        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                print("[ГРЕШКА] Не може да се прочете кадър.")
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            gesture = "none"
            cursor  = None
            grabbing= False

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            self.landmarker.detect_async(mp_image, timestamp_ms)
            timestamp_ms += int(1000 / CFG.fps_limit)

            result = self.latest_result

            if result and result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                draw_hand(frame, landmarks, w, h)
                gesture = detect_gesture(landmarks)
                ix, iy  = lm_xy(landmarks, INDEX_TIP)

                if self.tutorial_mode and self.tutorial:
                    self.tutorial.update(gesture)
                elif self.puzzle_mode and self.puzzle and not self.show_theme_menu:
                    cursor_x, cursor_y, grabbing = self.handle_puzzle(landmarks, w, h)
                    cursor = (cursor_x, cursor_y)
                    self.puzzle.update_hint_hover(cursor_x, cursor_y)
                    self.reset_scroll()
                elif self.presentation_mode and self.presentation:
                    cmd = self.presentation.update(gesture, ix)
                    if cmd:
                        self._pres_last_cmd = cmd
                        self._pres_last_t   = time.time()
                    self.reset_scroll()
                elif self.drawing_mode and self.drawing:
                    self.drawing.update(gesture, ix, iy, w, h)
                    self.reset_scroll()
                elif not self.puzzle_mode and not self.tutorial_mode:
                    if gesture == "move":
                        self.do_move(ix, iy); self.reset_scroll()
                        # Пусни drag ако бе активен
                        if self._drag_active:
                            pyautogui.mouseUp(button='left')
                            self._drag_active = False
                        self._pinch_held = False
                    elif gesture == "click":
                        # Drag / double-click логика
                        drag_res = self.update_drag(True, ix, iy)
                        gesture  = drag_res  # обновяваме за HUD
                        self.reset_scroll()
                    elif gesture == "right_click":
                        self.do_move(ix, iy); self.do_click("right"); self.reset_scroll()
                    elif gesture == "scroll":
                        self.do_scroll(iy)
                        if self._drag_active:
                            pyautogui.mouseUp(button='left'); self._drag_active = False
                        self._pinch_held = False
                    else:
                        self.update_drag(False, ix, iy)  # пускане на drag при none
                        self.reset_scroll()
            else:
                self.reset_scroll()
                # Ръката изчезна – пусни drag ако е активен
                if self._drag_active:
                    pyautogui.mouseUp(button='left')
                    self._drag_active = False
                self._pinch_held = False
                if self.puzzle_mode:
                    self._reset_pinch()
                    if self.pinch_prev and self.puzzle:
                        self.puzzle.release()

            # ── Рисуване ──
            if self.drawing_mode and self.drawing:
                self.drawing.draw(frame)
            if self.puzzle_mode and self.puzzle:
                self.puzzle.draw(frame, cursor, grabbing)

            draw_overlay(frame, gesture, self.fps, self.puzzle_mode, self.tutorial_mode,
                         self.presentation_mode, self.drawing_mode)

            if self.presentation_mode and self.presentation:
                pres_cmd = self._pres_last_cmd if time.time() - self._pres_last_t < 0.6 else None
                self.presentation.draw(frame, pres_cmd)

            self.settings_panel.draw(frame)

            if self.tutorial_mode and self.tutorial:
                self.tutorial.draw(frame, gesture,
                    result.hand_landmarks[0] if (result and result.hand_landmarks) else None)

            if self.show_theme_menu and self.selector:
                self.selector.draw(frame, self.current_theme_idx)

            # Модален прозорец за обучение (рисува се последен, отгоре)
            self._draw_tutorial_prompt(frame)

            # Начална страница – рисува се отгоре на всичко (когато е видима)
            self.home_screen.draw(frame, self.fps)

            self.update_fps()
            cv2.imshow(WINDOW_NAME, frame)

            # waitKeyEx дава пълен код на специалните клавиши (стрелки) на Windows
            key32 = cv2.waitKeyEx(1)
            key   = key32 & 0xFF

            # Началната страница поглъща клавишите (когато е видима)
            if self.home_screen.visible:
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == 27 or key == ord('h') or key == ord('H'):
                    self.home_screen.visible = False
                elif key in [ord(k.lower()) for k, *_ in HOME_MENU_ITEMS if k not in ('Q',)]:
                    # Препращаме клавиша към нормалната обработка след скриване на home
                    self.home_screen.visible = False
                    # fall-through към нормалните handler-и по-долу чрез повторна обработка
                    if key == ord('t'):
                        self._tut_prompt_mode = 'tutorial'; self._tut_prompt_active = True
                    elif key == ord('p'):
                        self._tut_prompt_mode = 'puzzle';   self._tut_prompt_active = True
                    elif key == ord('i'):
                        self._tut_prompt_mode = 'presentation'; self._tut_prompt_active = True
                    elif key == ord('w'):
                        self._tut_prompt_mode = 'drawing';  self._tut_prompt_active = True
                    elif key == ord('s'):
                        self.settings_panel.toggle()
                    elif key == ord('f'):
                        fs = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
                        if fs == cv2.WINDOW_FULLSCREEN:
                            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        else:
                            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                continue  # не обработваме другите handler-и

            # Модалният прозорец за обучение поглъща клавиши
            if self._tut_prompt_active:
                self._handle_tutorial_prompt_key(key, key32, w, h)
            elif key == ord("q"):
                break

            # H = покажи начална страница
            elif key == ord("h"):
                self.home_screen.toggle()
                print("[Home] " + ("Отворена" if self.home_screen.visible else "Затворена"))

            elif key == ord("t"):
                if self._tut_prompt_active:
                    pass
                elif self.tutorial_mode:
                    self.tutorial_mode = False
                    print("[Режим] Нормален")
                else:
                    self._tut_prompt_mode   = 'tutorial'
                    self._tut_prompt_active = True

            elif key == ord("p"):
                if self._tut_prompt_active:
                    pass
                elif self.puzzle_mode:
                    self.puzzle_mode = False
                    self.show_theme_menu = False
                    self._reset_pinch()
                    print("[Режим] Нормален")
                else:
                    self._tut_prompt_mode   = 'puzzle'
                    self._tut_prompt_active = True

            elif key == ord("n") and self.puzzle_mode:
                self._switch_theme(+1)

            elif key == ord("b") and self.puzzle_mode:
                self._switch_theme(-1)

            elif key == ord("m") and self.puzzle_mode:
                self.show_theme_menu = not self.show_theme_menu
                if self.show_theme_menu:
                    if self.selector is None or \
                       self.selector.fw != w or self.selector.fh != h:
                        self.selector = ThemeSelector(w, h)
                    print("[Меню] Избор на тема")
                else:
                    print("[Меню] Затворено")

            elif key == 13 and self.show_theme_menu:  # Enter
                self.show_theme_menu = False
                if self.puzzle:
                    self.puzzle.reset(theme_idx=self.current_theme_idx)
                    self._reset_pinch()
                theme = IMAGE_THEMES[self.current_theme_idx]
                print(f"[Тема] Избрана: {theme.label}")

            elif key == ord("r") and self.puzzle_mode and self.puzzle:
                self.puzzle.reset()
                self._reset_pinch()
                print("[Пъзел] Разбъркан отново")

            # D = смяна на трудност на пъзела
            elif key == ord("d") and self.puzzle_mode:
                idx = (DIFFICULTY_KEYS.index(self.puzzle_difficulty) + 1) % len(DIFFICULTY_KEYS)
                self.puzzle_difficulty = DIFFICULTY_KEYS[idx]
                if self.puzzle:
                    self.puzzle.reset(difficulty=self.puzzle_difficulty)
                    self._reset_pinch()
                print(f"[Пъзел] Трудност: {self.puzzle_difficulty}")

            # I = режим Презентация
            elif key == ord("i"):
                if self._tut_prompt_active:
                    pass
                elif self.presentation_mode:
                    self.presentation_mode = False
                    print("[Режим] Нормален")
                else:
                    self._tut_prompt_mode   = 'presentation'
                    self._tut_prompt_active = True

            # W = режим Рисуване
            elif key == ord("w"):
                if self._tut_prompt_active:
                    pass
                elif self.drawing_mode:
                    self.drawing_mode = False
                    print("[Режим] Нормален")
                else:
                    self._tut_prompt_mode   = 'drawing'
                    self._tut_prompt_active = True

            # C = изчисти canvas при рисуване
            elif key == ord("c") and self.drawing_mode and self.drawing:
                self.drawing.clear()
                print("[Рисуване] Изчистено")

            # Z = смяна дебелина при рисуване
            elif key == ord("z") and self.drawing_mode and self.drawing:
                self.drawing.change_thickness()
                t = DRAW_THICKNESS[self.drawing.thick_idx]
                print(f"[Рисуване] Дебелина: {t}px")

            # S = настройки
            elif key == ord("s"):
                self.settings_panel.toggle()
                print("[Настройки] " + ("Отворени" if self.settings_panel.visible else "Затворени"))

            # Навигация в настройките
            elif self.settings_panel.visible:
                # cv2.waitKeyEx на Windows: стрелки = 2490368/2621440/2424832/2555904
                # На Linux: 65362/65364/65361/65363 (горен байт)
                if key32 in (82, 65362, 2490368):    # Up
                    self.settings_panel.move(-1)
                elif key32 in (84, 65364, 2621440):  # Down
                    self.settings_panel.move(+1)
                elif key32 in (81, 65361, 2424832):  # Left
                    self.settings_panel.adjust(-1)
                elif key32 in (83, 65363, 2555904):  # Right
                    self.settings_panel.adjust(+1)
                elif key == 27:  # ESC
                    self.settings_panel.visible = False

            elif key == ord("f"):
                # F = toggle fullscreen
                fs = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
                if fs == cv2.WINDOW_FULLSCREEN:
                    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                else:
                    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            elapsed = time.time() - t0
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)

        cap.release()
        cv2.destroyAllWindows()
        self.landmarker.close()
        print("[OK] Спряно.")


# ──────────────────────────────────────────────
# Стартиране
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  CAMERA MOUSE CONTROL  (mediapipe 0.10+)")
    print("=" * 60)
    print("  ЖЕСТОВЕ (нормален режим):")
    print("    Показалец               -> Движение на мишката")
    print("    Пинч (палец+показалец)  -> Ляв клик")
    print("    Показалец + среден      -> Скрол нагоре/надолу")
    print("    Показалец+среден+безим. -> Десен клик")
    print()
    print("  ПЪЗЕЛ РЕЖИМ (P):")
    print("    8 вградени теми: Океан, Залез, Гора, Космос,")
    print("                     Бонбони, Планини, Абстракт, Плаж")
    print("    N = следваща тема    B = предишна тема")
    print("    M = визуално меню    R = разбъркай")
    print()
    print("  УРОК РЕЖИМ: T     FULLSCREEN: F     ИЗХОД: Q")
    print()
    print("  НОВИ РЕЖИМИ:")
    print("    I = Презентация (свайп ляво/дясно = слайдове)")
    print("    W = Рисуване (показалец рисува, 2 пръста = цвят, Z = дебелина, C = изчисти)")
    print("    S = Настройки (стрелки за навигация и промяна)")
    print()
    print("  ЖЕСТ ПОДОБРЕНИЯ:")
    print("    Двоен бърз пинч                 -> Двоен клик")
    print("    Задържан пинч (>0.55 сек)        -> Drag & Drop")
    print()
    print("  ПЪЗЕЛ: D = трудност (2x2 / 3x3 / 4x4)  Рекорди в puzzle_records.json")
    print("         Задръж курсор 2 сек над парче -> Подсказка")
    print("=" * 60)
    print()

    try:
        CameraMouseController().run()
    except pyautogui.FailSafeException:
        print("\n[STOP] Fail-safe задействан!")
    except KeyboardInterrupt:
        print("\n[STOP] Прекъснато.")