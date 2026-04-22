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
import requests
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

from PIL import Image, ImageDraw, ImageFont

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
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    font    = get_font(font_size)
    draw.text((pos[0]+1, pos[1]+1), text, font=font, fill=(0,0,0,180))
    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
    result  = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    np.copyto(frame, result)

# ──────────────────────────────────────────────
# Вградени снимки – генерираме ги процедурно
# (няма нужда от файлове на диска)
# ──────────────────────────────────────────────

def _make_gradient_image(w: int, h: int,
                          c1: Tuple, c2: Tuple, c3: Tuple,
                          style: str = "diagonal") -> np.ndarray:
    """Генерира красиво градиентно изображение."""
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


def _add_noise(img: np.ndarray, amount: float = 8.0) -> np.ndarray:
    noise = np.random.randn(*img.shape) * amount
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _add_circles(img: np.ndarray, color: Tuple, count: int = 12) -> np.ndarray:
    h, w = img.shape[:2]
    out  = img.copy()
    for _ in range(count):
        cx  = random.randint(0, w)
        cy  = random.randint(0, h)
        r   = random.randint(20, min(w, h) // 3)
        alpha = random.uniform(0.08, 0.22)
        overlay = out.copy()
        cv2.circle(overlay, (cx, cy), r, color, -1)
        cv2.addWeighted(overlay, alpha, out, 1-alpha, 0, out)
    return out


def _add_stars(img: np.ndarray, count: int = 60) -> np.ndarray:
    h, w = img.shape[:2]
    out  = img.copy()
    for _ in range(count):
        x = random.randint(0, w-1)
        y = random.randint(0, h-1)
        r = random.randint(1, 3)
        brightness = random.randint(180, 255)
        cv2.circle(out, (x, y), r, (brightness, brightness, brightness), -1)
    return out


def _add_waves(img: np.ndarray, color: Tuple, count: int = 5) -> np.ndarray:
    h, w = img.shape[:2]
    out  = img.copy()
    for i in range(count):
        pts = []
        amp    = random.randint(15, 40)
        freq   = random.uniform(0.01, 0.04)
        offset = random.randint(50, h - 50)
        for x in range(0, w, 4):
            y = int(offset + amp * math.sin(freq * x + i))
            pts.append((x, max(0, min(h-1, y))))
        for j in range(len(pts)-1):
            alpha = random.uniform(0.1, 0.25)
            overlay = out.copy()
            cv2.line(overlay, pts[j], pts[j+1], color, 2)
            cv2.addWeighted(overlay, alpha, out, 1-alpha, 0, out)
    return out


def _add_triangles(img: np.ndarray, color: Tuple, count: int = 8) -> np.ndarray:
    h, w = img.shape[:2]
    out  = img.copy()
    for _ in range(count):
        pts = np.array([
            [random.randint(0, w), random.randint(0, h)],
            [random.randint(0, w), random.randint(0, h)],
            [random.randint(0, w), random.randint(0, h)],
        ], np.int32)
        alpha = random.uniform(0.06, 0.18)
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, alpha, out, 1-alpha, 0, out)
    return out


def _add_label(img: np.ndarray, text: str, emoji: str) -> np.ndarray:
    """Добавя надпис с емоджи в центъра на изображението."""
    h, w = img.shape[:2]
    # Полупрозрачен правоъгълник
    overlay = img.copy()
    tw, th  = w // 3, h // 8
    x1      = w // 2 - tw // 2
    y1      = h // 2 - th // 2
    cv2.rectangle(overlay, (x1, y1), (x1+tw, y1+th), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    font_big = get_font(32)
    font_sm  = get_font(18)
    # Емоджи горе
    draw.text((w//2 - 20, y1 - 40), emoji, font=font_big, fill=(255,255,255))
    # Текст
    draw.text((x1 + 8, y1 + 6), text, font=font_sm, fill=(255,255,255))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ── Дефиниции на темите ──────────────────────

@dataclass
class ImageTheme:
    key:   str
    label: str
    emoji: str
    image: Optional[np.ndarray] = field(default=None, repr=False)

    def build(self, w: int, h: int) -> None:
        self.image = _generate_theme_image(self.key, w, h)


def _generate_theme_image(key: str, w: int, h: int) -> np.ndarray:
    random.seed(hash(key) % 9999)  # Винаги еднакво за една тема

    if key == "ocean":
        img = Image.open("puzzle_images/ocean.jpg").convert("RGB")
        img = img.resize((w, h))

        # PIL -> numpy (ВАЖНО)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        img = _add_noise(img, 6)
        img = _add_waves(img, (100, 210, 230), count=8)
        img = _add_circles(img, (50, 180, 220), count=5)

    elif key == "sunset":
        img = cv2.imread("puzzle_images/sunset.jpg")
        img = cv2.resize(img, (w, h))
        img = _add_noise(img, 5)
        cv2.circle(img, (int(w * 0.65), int(h * 0.35)), 55, (0, 200, 255), -1)
        cv2.circle(img, (int(w * 0.65), int(h * 0.35)), 60, (0, 170, 230), 4)
        img = _add_triangles(img, (0, 80, 200), count=10)

    elif key == "forest":
        img = cv2.imread("puzzle_images/forest.jpg")
        img = cv2.resize(img, (w, h))

        img = _add_noise(img, 8)
        img = _add_circles(img, (60, 180, 40), count=15)
        img = _add_triangles(img, (20, 100, 20), count=12)

    elif key == "space":
        img = cv2.imread("puzzle_images/space.jpg")
        img = cv2.resize(img, (w, h))

        img = _add_noise(img, 4)
        img = _add_stars(img, count=120)

        cx, cy = int(w*0.7), int(h*0.35)
        cv2.circle(img, (cx, cy), 70, (60, 30, 120), -1)
        cv2.circle(img, (cx, cy), 70, (100, 60, 180), 4)
        cv2.ellipse(img, (cx, cy), (110, 30), -20, 0, 360, (80, 50, 140), 3)

    elif key == "candy":
        img = cv2.imread("puzzle_images/candy.jpg")
        img = cv2.resize(img, (w, h))

        img = _add_noise(img, 7)
        img = _add_circles(img, (255, 200, 60), count=12)
        img = _add_triangles(img, (200, 80, 240), count=8)

        for _ in range(25):
            cx_ = random.randint(0, w)
            cy_ = random.randint(0, h)
            r_  = random.randint(5, 18)
            col = (random.randint(200,255), random.randint(100,200), random.randint(150,255))
            cv2.circle(img, (cx_, cy_), r_, col, -1)

    elif key == "mountain":
        img = cv2.imread("puzzle_images/mountain.jpg")
        img = cv2.resize(img, (w, h))

        img = _add_noise(img, 6)

    elif key == "abstract":
        img = cv2.imread("puzzle_images/abstract.jpg")
        img = cv2.resize(img, (w, h))

        img = _add_noise(img, 10)
        img = _add_triangles(img, (200, 80, 240), count=20)
        img = _add_circles(img, (255, 120, 60), count=10)

    elif key == "beach":
        img = cv2.imread("puzzle_images/beach.jpg")
        img = cv2.resize(img, (w, h))

        img = _add_noise(img, 5)
        img = _add_waves(img, (80, 200, 220), count=4)
        cv2.circle(img, (int(w*0.75), int(h*0.2)), 45, (0, 210, 255), -1)

    else:
        img = _make_gradient_image(
            w, h,
            (80, 80, 80), (160, 160, 180), (40, 40, 60),
            style="diagonal"
        )
        img = _add_noise(img, 5)

    return img


IMAGE_THEMES: List[ImageTheme] = [
    ImageTheme("ocean",    "Океан",    "🌊"),
    ImageTheme("sunset",   "Залез",    "🌅"),
    ImageTheme("forest",   "Гора",     "🌲"),
    ImageTheme("space",    "Космос",   "🌌"),
    ImageTheme("candy",    "Бонбони",  "🍭"),
    ImageTheme("mountain", "Планини",  "⛰"),
    ImageTheme("abstract", "Абстракт", "🎨"),
    ImageTheme("beach",    "Плаж",     "🏖"),
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
    cam_width:      int   = 640
    cam_height:     int   = 480
    fps_limit:      int   = 30
    smooth_window:  int   = 6
    active_zone_x:  Tuple = (0.1, 0.9)
    active_zone_y:  Tuple = (0.1, 0.9)
    click_distance: float = 0.05
    click_cooldown: float = 0.4
    scroll_sens:    float = 400.0
    scroll_dead:    float = 0.008

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
# Туториал режим
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
    put_text_unicode(frame, "☝", (cx-12, cy-55), font_size=32, color=(255,220,80))

def _icon_pinch(frame, cx, cy):
    cv2.circle(frame, (cx-15, cy),    9, (255,255,255), -1)
    cv2.circle(frame, (cx+15, cy),    9, (255,255,255), -1)
    cv2.line(frame,   (cx-6, cy), (cx+6, cy), (0,180,255), 2)
    put_text_unicode(frame, "🤏", (cx-18, cy-55), font_size=30, color=(0,180,255))

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
        alpha = 0.72
        overlay = frame.copy()
        panel_w, panel_h = 300, 240
        px, py = 8, h - panel_h - 8
        draw_rounded_rect(overlay, px, py, px+panel_w, py+panel_h, TUTORIAL_BG, radius=14)
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

        if self.all_done:
            put_text_unicode(frame, "Туториалът е завършен!", (px+14, py+90),
                             font_size=18, color=(0,255,120))
            put_text_unicode(frame, "Натиснете T за изход", (px+14, py+122),
                             font_size=15, color=(180,180,180))
            return

        step = self.current
        put_text_unicode(frame, step.title, (px+12, py+10), font_size=15, color=(0,200,255))
        for i, line in enumerate(step.description.split("\n")):
            put_text_unicode(frame, line, (px+12, py+34+i*20), font_size=13, color=(210,210,210))

        bar_x, bar_y = px+12, py+panel_h-46
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
        put_text_unicode(frame, status, (px+12, py+panel_h-26), font_size=13, color=color)

        total = len(TUTORIAL_STEPS)
        for i in range(total):
            cx = px + 12 + i * 26
            cy = py + panel_h - 60
            col = TUTORIAL_OK if i < self.step_idx else \
                  TUTORIAL_ACNT if i == self.step_idx else (60,60,80)
            cv2.circle(frame, (cx, cy), 7, col, -1)

        step.icon_func(frame, px + panel_w - 50, py + 85)


# ──────────────────────────────────────────────
# Избор на тема – красив менюселектор
# ──────────────────────────────────────────────

class ThemeSelector:
    """Показва миниатюри на темите и позволява избор с N/B."""

    THUMB_W = 100
    THUMB_H =  70
    COLS    =   4
    PAD     =   8

    def __init__(self, frame_w: int, frame_h: int):
        self.fw = frame_w
        self.fh = frame_h
        self.thumbs: List[np.ndarray] = []
        self._build_thumbs(frame_w, frame_h)

    def _build_thumbs(self, fw: int, fh: int):
        # Генерираме пъзел изображения (по-малък размер за миниатюри)
        bw = fw // 2
        bh = fh
        self.thumbs = []
        for theme in IMAGE_THEMES:
            if theme.image is None:
                theme.build(bw, bh)
            thumb = cv2.resize(theme.image, (self.THUMB_W, self.THUMB_H))
            self.thumbs.append(thumb)

    def draw(self, frame: np.ndarray, current_idx: int) -> None:
        h, w = frame.shape[:2]
        rows = math.ceil(len(IMAGE_THEMES) / self.COLS)
        total_w = self.COLS * (self.THUMB_W + self.PAD) + self.PAD
        total_h = rows * (self.THUMB_H + self.PAD) + self.PAD + 40

        start_x = w // 2 - total_w // 2
        start_y = h // 2 - total_h // 2

        # Фон
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
            y   = start_y + 36 + row * (self.THUMB_H + self.PAD)

            # Миниатюра
            frame[y:y+self.THUMB_H, x:x+self.THUMB_W] = self.thumbs[i]

            # Рамка (по-дебела за избраната)
            if i == current_idx:
                cv2.rectangle(frame, (x-3, y-3),
                              (x+self.THUMB_W+3, y+self.THUMB_H+3),
                              (0, 220, 255), 3)
                # Стрелка горе
                put_text_unicode(frame, "▼", (x + self.THUMB_W//2 - 6, y - 20),
                                 font_size=14, color=(0, 220, 255))
            else:
                cv2.rectangle(frame, (x, y), (x+self.THUMB_W, y+self.THUMB_H),
                              (80, 80, 100), 1)

            # Надпис
            put_text_unicode(frame, theme.emoji + " " + theme.label,
                             (x + 2, y + self.THUMB_H + 2),
                             font_size=11, color=(200,200,220))


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


class PuzzleGame:
    COLS      = 3
    ROWS      = 3
    SNAP_DIST = 32

    def __init__(self, frame_w: int, frame_h: int, theme_idx: int = 0):
        self.fw        = frame_w
        self.fh        = frame_h
        self.theme_idx = theme_idx

        self.board_x = frame_w // 2
        self.board_y = 0
        self.board_w = frame_w // 2
        self.board_h = frame_h

        self.pieces:    List[PuzzlePiece] = []
        self.held:      Optional[int]     = None
        self.hold_off_x = 0
        self.hold_off_y = 0
        self.completed  = False
        self.start_time = time.time()
        self.end_time:  Optional[float] = None
        self.snap_anim: dict = {}

        self._build_puzzle()

    def _get_source(self) -> np.ndarray:
        theme = IMAGE_THEMES[self.theme_idx % len(IMAGE_THEMES)]
        if theme.image is None:
            theme.build(self.board_w, self.board_h)
        return cv2.resize(theme.image, (self.board_w, self.board_h))

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

                # Решетка
                cv2.rectangle(piece_img, (0,0), (pw-1, ph-1), (255,255,255), 2)

                # Номер
                num = r * self.COLS + c + 1
                img_pil = Image.fromarray(cv2.cvtColor(piece_img, cv2.COLOR_BGR2RGB))
                draw    = ImageDraw.Draw(img_pil)
                font    = get_font(20)
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
            self.completed = True
            self.end_time  = time.time()

    def reset(self, theme_idx: Optional[int] = None):
        if theme_idx is not None:
            self.theme_idx = theme_idx
        self.completed  = False
        self.end_time   = None
        self.held       = None
        self.snap_anim  = {}
        self.start_time = time.time()
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

        # Разделителна линия
        cv2.line(frame, (self.board_x, 0), (self.board_x, h), (60, 100, 140), 2)

        # Прогрес панел
        placed  = sum(1 for p in self.pieces if p.placed)
        total   = len(self.pieces)
        elapsed = int(now - self.start_time)
        panel_x = self.board_x + 4

        draw_rounded_rect(frame, panel_x, 4, panel_x + 210, 72, (20, 20, 35), radius=8)
        put_text_unicode(frame,
            f"{theme.emoji} {theme.label}: {placed}/{total}",
            (panel_x + 8, 8), font_size=14, color=(0, 200, 255))
        put_text_unicode(frame, f"Време: {elapsed} сек",
            (panel_x + 8, 30), font_size=13, color=(160, 200, 160))

        bar_x, bar_y = panel_x + 8, 54
        bar_w = 194
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+10), (40,40,60), -1)
        fill_w = int(bar_w * placed / total) if total > 0 else 0
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x+fill_w, bar_y+10), (0,200,100), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+10), (80,80,100), 1)

        # Инструкции
        hints_y = h - 76
        draw_rounded_rect(frame, panel_x, hints_y, panel_x + 210, h - 4, (20,20,35), radius=8)
        put_text_unicode(frame, "N = следваща тема",   (panel_x+8, hints_y+6),  font_size=12, color=(140,140,160))
        put_text_unicode(frame, "B = предишна тема",   (panel_x+8, hints_y+22), font_size=12, color=(140,140,160))
        put_text_unicode(frame, "R = разбъркай",       (panel_x+8, hints_y+38), font_size=12, color=(140,140,160))
        put_text_unicode(frame, "M = меню с теми",     (panel_x+8, hints_y+54), font_size=12, color=(140,140,160))

        # Курсор
        if cursor_xy:
            cx, cy = cursor_xy
            pulse_c = int(200 + 55 * math.sin(now * 6))
            color   = (0, pulse_c, 255) if is_grabbing else (255, 255, 255)
            cv2.circle(frame, (cx, cy), 14, color, 2)
            cv2.circle(frame, (cx, cy),  3, color, -1)
            cv2.line(frame, (cx-20, cy), (cx-8, cy),  color, 1)
            cv2.line(frame, (cx+8,  cy), (cx+20, cy), color, 1)
            cv2.line(frame, (cx, cy-20), (cx, cy-8),  color, 1)
            cv2.line(frame, (cx, cy+8),  (cx, cy+20), color, 1)
            if is_grabbing:
                put_text_unicode(frame, "ХВАНАТО", (cx+18, cy-12),
                                 font_size=13, color=(0, 220, 255))

        # Победа
        if self.completed and self.end_time:
            self._draw_win_banner(frame, int(self.end_time - self.start_time))

    def _draw_win_banner(self, frame: np.ndarray, duration: int):
        h, w = frame.shape[:2]
        theme = IMAGE_THEMES[self.theme_idx % len(IMAGE_THEMES)]
        overlay = frame.copy()
        bx1, by1 = w//4 - 20, h//3 - 10
        bx2, by2 = 3*w//4 + 20, 2*h//3 + 20
        draw_rounded_rect(overlay, bx1, by1, bx2, by2, (10, 50, 20), radius=16)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        draw_rounded_rect(frame, bx1, by1, bx2, by2, (0, 200, 80), radius=16, thickness=3)
        put_text_unicode(frame, "БРАВО!",
            (bx1 + 60, by1 + 20), font_size=28, color=(0, 255, 120))
        put_text_unicode(frame, f"Завършен за {duration} секунди",
            (bx1 + 30, by1 + 70), font_size=18, color=(200, 240, 200))
        put_text_unicode(frame, f"Тема: {theme.emoji} {theme.label}",
            (bx1 + 50, by1 + 100), font_size=15, color=(160, 200, 160))
        put_text_unicode(frame, "N = нова тема  |  R = пак",
            (bx1 + 40, by1 + 130), font_size=14, color=(120, 160, 120))


# ──────────────────────────────────────────────
# HUD
# ──────────────────────────────────────────────

GESTURE_LABEL = {
    "move":        "Движение",
    "click":       "Пинч клик",
    "right_click": "Десен клик",
    "scroll":      "Скрол",
    "none":        "—",
}
GESTURE_COLOR = {
    "move":        (0,255,80),
    "click":       (0,120,255),
    "right_click": (255,100,0),
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
                 puzzle_mode: bool, tutorial_mode: bool) -> None:
    h, w = frame.shape[:2]

    if not puzzle_mode and not tutorial_mode:
        ax0 = int(CFG.active_zone_x[0] * w)
        ax1 = int(CFG.active_zone_x[1] * w)
        ay0 = int(CFG.active_zone_y[0] * h)
        ay1 = int(CFG.active_zone_y[1] * h)
        cv2.rectangle(frame, (ax0, ay0), (ax1, ay1), (0, 200, 80), 1)

    draw_rounded_rect(frame, 6, 4, 246, 70, (15, 15, 35), radius=8)
    color = GESTURE_COLOR.get(gesture, (160,160,160))

    mode_str = ""
    if puzzle_mode:   mode_str = "[ПЪЗЕЛ] "
    if tutorial_mode: mode_str = "[УРОК] "

    put_text_unicode(frame, f"{mode_str}Жест: {GESTURE_LABEL.get(gesture,'?')}",
                     (12, 8), font_size=15, color=color)
    put_text_unicode(frame, f"FPS: {fps:.1f}  |  T=Урок  P=Пъзел  Q=Изход",
                     (12, 32), font_size=12, color=(160, 160, 160))

    if not puzzle_mode and not tutorial_mode:
        hints = [
            ("Показалец",  "Движение"),
            ("Пинч",       "Ляв клик"),
            ("2 пръста",   "Скрол"),
            ("3 пръста",   "Десен клик"),
            ("T",          "Урок режим"),
            ("P",          "Пъзел режим"),
        ]
        panel_x = w - 220
        draw_rounded_rect(frame, panel_x-4, 4, w-4, 4+len(hints)*22+8, (15,15,35), radius=8)
        for i, (g, a) in enumerate(hints):
            put_text_unicode(frame, f"{g}: {a}", (panel_x, 10+i*22),
                             font_size=12, color=(150,180,150))


# ──────────────────────────────────────────────
# Главен контролер
# ──────────────────────────────────────────────

class CameraMouseController:
    def __init__(self):
        download_model()
        ensure_font()

        # Предгенерираме всички теми (малки изображения) за по-бърз старт
        print("Генерирам теми за пъзела...", end=" ", flush=True)
        for theme in IMAGE_THEMES:
            theme.build(320, 240)   # ще се регенерират при реален размер при нужда
        print("OK")

        self.pos_buffer      = deque(maxlen=CFG.smooth_window)
        self.last_click      = 0.0
        self.scroll_ref_y:   Optional[float] = None
        self.scroll_accum:   float = 0.0
        self.latest_result:  Optional[HandLandmarkerResult] = None
        self.fps             = 0.0
        self._frame_count    = 0
        self._fps_timer      = time.time()

        self.puzzle_mode    = False
        self.tutorial_mode  = False
        self.show_theme_menu= False
        self.puzzle:        Optional[PuzzleGame]    = None
        self.tutorial:      Optional[TutorialMode]  = None
        self.selector:      Optional[ThemeSelector] = None
        self.current_theme_idx = 0

        self.PINCH_CONFIRM = 3
        self.pinch_raw_buf = deque(maxlen=self.PINCH_CONFIRM)
        self.pinch_stable  = False
        self.pinch_prev    = False
        self.puzzle_cursor_buf = deque(maxlen=8)

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
        now = time.time()
        if now - self.last_click >= CFG.click_cooldown:
            pyautogui.click(button=button)
            self.last_click = now

    def do_scroll(self, ny: float) -> None:
        if self.scroll_ref_y is None:
            self.scroll_ref_y = ny
            self.scroll_accum = 0.0
            return
        delta_norm     = self.scroll_ref_y - ny
        self.scroll_accum += delta_norm * SCREEN_H
        scroll_units   = int(self.scroll_accum / CFG.scroll_sens)
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
        """Сменя темата с delta стъпки (+1 или -1)."""
        self.current_theme_idx = (self.current_theme_idx + delta) % len(IMAGE_THEMES)
        if self.puzzle:
            self.puzzle.reset(theme_idx=self.current_theme_idx)
            self._reset_pinch()
        theme = IMAGE_THEMES[self.current_theme_idx]
        print(f"[Тема] {theme.emoji} {theme.label}")

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

            frame   = cv2.flip(frame, 1)
            h, w    = frame.shape[:2]
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
                    self.reset_scroll()
                elif not self.puzzle_mode and not self.tutorial_mode:
                    if gesture == "move":
                        self.do_move(ix, iy); self.reset_scroll()
                    elif gesture == "click":
                        self.do_move(ix, iy); self.do_click("left"); self.reset_scroll()
                    elif gesture == "right_click":
                        self.do_move(ix, iy); self.do_click("right"); self.reset_scroll()
                    elif gesture == "scroll":
                        self.do_scroll(iy)
                    else:
                        self.reset_scroll()
            else:
                self.reset_scroll()
                if self.puzzle_mode:
                    self._reset_pinch()
                    if self.pinch_prev and self.puzzle:
                        self.puzzle.release()

            # ── Рисуване ──
            if self.puzzle_mode and self.puzzle:
                self.puzzle.draw(frame, cursor, grabbing)

            draw_overlay(frame, gesture, self.fps, self.puzzle_mode, self.tutorial_mode)

            if self.tutorial_mode and self.tutorial:
                self.tutorial.draw(frame, gesture,
                    result.hand_landmarks[0] if (result and result.hand_landmarks) else None)

            # Меню за теми (наслагване върху всичко останало)
            if self.show_theme_menu and self.selector:
                self.selector.draw(frame, self.current_theme_idx)

            self.update_fps()
            cv2.imshow("Camera Mouse Control", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            elif key == ord("t"):
                self.tutorial_mode  = not self.tutorial_mode
                self.puzzle_mode    = False
                self.show_theme_menu= False
                if self.tutorial_mode:
                    self.tutorial = TutorialMode()
                    print("[Режим] Урок")
                else:
                    print("[Режим] Нормален")

            elif key == ord("p"):
                self.tutorial_mode  = False
                self.show_theme_menu= False
                self.puzzle_mode    = not self.puzzle_mode
                self._reset_pinch()
                if self.puzzle_mode:
                    if self.puzzle is None:
                        self.puzzle = PuzzleGame(w, h, theme_idx=self.current_theme_idx)
                    elif self.puzzle.completed:
                        self.puzzle.reset(theme_idx=self.current_theme_idx)
                    theme = IMAGE_THEMES[self.current_theme_idx]
                    print(f"[Режим] Пъзел – {theme.emoji} {theme.label}")
                else:
                    print("[Режим] Нормален")

            elif key == ord("n") and self.puzzle_mode:
                self._switch_theme(+1)

            elif key == ord("b") and self.puzzle_mode:
                self._switch_theme(-1)

            elif key == ord("m") and self.puzzle_mode:
                self.show_theme_menu = not self.show_theme_menu
                if self.show_theme_menu:
                    if self.selector is None:
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
                print(f"[Тема] Избрана: {theme.emoji} {theme.label}")

            elif key == ord("r") and self.puzzle_mode and self.puzzle:
                self.puzzle.reset()
                self._reset_pinch()
                print("[Пъзел] Разбъркан отново")

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
    print("  УРОК РЕЖИМ: T     ИЗХОД: Q")
    print("=" * 60)
    print()

    try:
        CameraMouseController().run()
    except pyautogui.FailSafeException:
        print("\n[STOP] Fail-safe задействан!")
    except KeyboardInterrupt:
        print("\n[STOP] Прекъснато.")