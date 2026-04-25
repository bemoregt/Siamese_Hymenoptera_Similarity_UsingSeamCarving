"""
Siamese Network 유사도 측정 GUI — Windows 2000 Classic Style
"""

import tkinter as tk
from tkinter import filedialog
import random
import glob
import os
import sys
from PIL import Image, ImageTk
import torch
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from siamese_train import SiameseNet

# ── 경로 / 설정 ──────────────────────────────────────────────────────────────
BASE      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hymenoptera")
CKPT      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "best_siamese.pth")
DISP_SZ   = 240
THRESHOLD = 0.5      # cosine sim > THRESHOLD → same class

# ── Windows 2000 Classic 팔레트 ───────────────────────────────────────────────
W2K = dict(
    bg       = "#d4d0c8",
    btn      = "#d4d0c8",
    btn_act  = "#ece9d8",
    white    = "#ffffff",
    black    = "#000000",
    gray_dk  = "#808080",
    gray_dkr = "#404040",
    titlebar = "#0a246a",
    title_fg = "#ffffff",
    pos      = "#007a00",
    neg      = "#bb0000",
    bar_bg   = "#808080",
    bar_pos  = "#007a00",
    bar_neg  = "#bb0000",
    bar_off  = "#c0c0c0",
)

# ── 폰트 ─────────────────────────────────────────────────────────────────────
def _font(*args):
    # macOS 에서도 쓸 수 있는 폴백 폰트
    for fam in ("MS Sans Serif", "Arial", "Helvetica"):
        try:
            f = tk.font.Font(family=fam, size=args[0] if args else 8,
                             weight=args[1] if len(args) > 1 else "normal")
            return (fam,) + args
        except Exception:
            pass
    return ("Helvetica",) + args

# ── 모델 ─────────────────────────────────────────────────────────────────────
_TF = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def load_model():
    m = SiameseNet(embed_dim=128)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model"])
    m.eval()
    return m

def cosine_sim(model, p1, p2):
    def emb(p):
        t = _TF(Image.open(p).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            return model.embed(t)
    return (emb(p1) * emb(p2)).sum().item()

# ── 이미지 목록 수집 ──────────────────────────────────────────────────────────
def collect():
    pool = {}
    for split in ("train", "val"):
        for cls in ("ants", "bees"):
            files = sorted(glob.glob(os.path.join(BASE, split, cls, "*.jpg")))
            pool.setdefault(cls, []).extend(files)
    return pool   # {"ants": [...], "bees": [...]}

# ── 커스텀 위젯 ───────────────────────────────────────────────────────────────

class W2kButton(tk.Button):
    def __init__(self, master, **kw):
        kw.setdefault("bg",             W2K["btn"])
        kw.setdefault("activebackground", W2K["btn_act"])
        kw.setdefault("relief",         "raised")
        kw.setdefault("borderwidth",    2)
        kw.setdefault("cursor",         "arrow")
        kw.setdefault("padx",           6)
        kw.setdefault("pady",           3)
        super().__init__(master, **kw)


class SegBar(tk.Canvas):
    """세그먼트형 Windows 2000 프로그레스 바"""
    SEG_W = 11
    SEG_H = 14
    GAP   = 2

    def __init__(self, master, width=440, **kw):
        self._bar_w = width
        super().__init__(master, width=width, height=self.SEG_H + 4,
                         bg=W2K["bg"], relief="sunken", borderwidth=2,
                         highlightthickness=0, **kw)
        self.set(0.0, positive=True)

    def set(self, value, positive=True):
        self.delete("all")
        n = (self._bar_w - 4) // (self.SEG_W + self.GAP)
        filled = int(n * max(0.0, min(1.0, value)))
        on  = W2K["bar_pos"] if positive else W2K["bar_neg"]
        off = W2K["bar_off"]
        for i in range(n):
            x0 = 2 + i * (self.SEG_W + self.GAP)
            self.create_rectangle(x0, 2, x0 + self.SEG_W, self.SEG_H + 2,
                                  fill=(on if i < filled else off),
                                  outline="")


# ── 메인 앱 ───────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Siamese 유사도 측정기")
        self.resizable(False, False)
        self.configure(bg=W2K["bg"])

        # 임포트 확인 후 폰트 모듈 사용
        import tkinter.font as tkfont
        self._tkfont = tkfont

        self._fonts = {
            "n":  _font(9),
            "b":  _font(9,  "bold"),
            "l":  _font(11, "bold"),
            "xl": _font(22, "bold"),
            "s":  _font(8),
            "ss": _font(7),
        }

        self.model  = load_model()
        self.pool   = collect()
        self.all_imgs = [p for v in self.pool.values() for p in v]

        self.paths  = [None, None]
        self.cls    = [None, None]
        self.photos = [None, None]

        self._build()
        for i in range(2):
            self._load(i)
        self._compute()

    # ── UI 빌드 ───────────────────────────────────────────────────────────────

    def _build(self):
        # 타이틀 바 시뮬레이션
        tb = tk.Frame(self, bg=W2K["titlebar"], height=22)
        tb.pack(fill="x")
        tk.Label(tb, text="  Siamese Network - 이미지 유사도 측정기",
                 bg=W2K["titlebar"], fg=W2K["title_fg"],
                 font=self._fonts["b"], anchor="w").pack(side="left", fill="y")

        # 메인 컨테이너
        body = tk.Frame(self, bg=W2K["bg"], padx=10, pady=8)
        body.pack(fill="both", expand=True)

        # 이미지 패널 2개
        top = tk.Frame(body, bg=W2K["bg"])
        top.pack()
        for i in range(2):
            self._build_panel(top, i).grid(row=0, column=i, padx=8, pady=4)

        # 구분선
        tk.Frame(body, bg=W2K["gray_dk"], height=1).pack(fill="x", pady=4)

        # 결과 박스
        rb = tk.Frame(body, bg=W2K["bg"], relief="raised", borderwidth=2)
        rb.pack(fill="x", padx=2, pady=2)
        ri = tk.Frame(rb, bg=W2K["bg"], padx=10, pady=8)
        ri.pack(fill="x")

        # 점수 행
        sr = tk.Frame(ri, bg=W2K["bg"])
        sr.pack(fill="x")
        tk.Label(sr, text="유사도 점수", bg=W2K["bg"],
                 font=self._fonts["n"], fg=W2K["gray_dkr"]).pack(side="left")
        self.lbl_score = tk.Label(sr, text="---", bg=W2K["bg"],
                                  font=self._fonts["xl"], fg=W2K["black"],
                                  width=8, anchor="e")
        self.lbl_score.pack(side="right")

        # 세그먼트 바
        self.bar = SegBar(ri, width=476)
        self.bar.pack(fill="x", pady=4)

        # 판정
        self.lbl_verdict = tk.Label(ri, text="",
                                    bg=W2K["bg"], font=self._fonts["l"],
                                    anchor="center")
        self.lbl_verdict.pack(fill="x")

        # 하단 정보 행
        info_row = tk.Frame(ri, bg=W2K["bg"])
        info_row.pack(fill="x", pady=(4, 0))
        tk.Label(info_row, text="cosine similarity :", bg=W2K["bg"],
                 font=self._fonts["s"], fg=W2K["gray_dkr"]).pack(side="left")
        self.lbl_raw = tk.Label(info_row, text="---", bg=W2K["bg"],
                                font=self._fonts["s"], fg=W2K["gray_dkr"])
        self.lbl_raw.pack(side="left", padx=4)
        tk.Label(info_row, text=f"  임계값 : {THRESHOLD:.2f}",
                 bg=W2K["bg"], font=self._fonts["s"],
                 fg=W2K["gray_dkr"]).pack(side="left")

        # 상태바
        tk.Frame(body, bg=W2K["gray_dk"], height=1).pack(fill="x", pady=(6, 2))
        self.lbl_status = tk.Label(body, text="준비", bg=W2K["bg"],
                                   font=self._fonts["s"],
                                   relief="sunken", borderwidth=1,
                                   anchor="w", padx=4)
        self.lbl_status.pack(fill="x")

    def _build_panel(self, parent, idx):
        outer = tk.Frame(parent, bg=W2K["bg"], relief="raised", borderwidth=2)

        # 패널 헤더
        hdr = tk.Frame(outer, bg=W2K["titlebar"], height=18)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text=f"  이미지 {idx + 1}",
                 bg=W2K["titlebar"], fg=W2K["title_fg"],
                 font=self._fonts["b"], anchor="w").pack(fill="both")

        inner = tk.Frame(outer, bg=W2K["bg"], padx=6, pady=6)
        inner.pack()

        # 이미지 표시 (sunken 테두리)
        img_border = tk.Frame(inner, bg=W2K["white"],
                              relief="sunken", borderwidth=2)
        img_border.pack()
        lbl_img = tk.Label(img_border, bg=W2K["white"],
                           width=DISP_SZ, height=DISP_SZ)
        lbl_img.pack()
        setattr(self, f"_lbl_img{idx}", lbl_img)

        # 클래스 레이블 (sunken)
        lbl_cls = tk.Label(inner, text="클래스 : ---",
                           bg=W2K["bg"], font=self._fonts["b"],
                           relief="sunken", borderwidth=1,
                           padx=4, pady=2, anchor="w")
        lbl_cls.pack(fill="x", pady=(4, 2))
        setattr(self, f"_lbl_cls{idx}", lbl_cls)

        # 파일명 (작은 글씨)
        lbl_fn = tk.Label(inner, text="", bg=W2K["bg"],
                          font=self._fonts["ss"],
                          fg=W2K["gray_dk"], anchor="w")
        lbl_fn.pack(fill="x")
        setattr(self, f"_lbl_fn{idx}", lbl_fn)

        # 버튼 — 좌측: 랜덤, 우측: 파일 대화창
        if idx == 0:
            W2kButton(inner, text="  >> 랜덤 이미지 불러오기",
                      font=self._fonts["b"],
                      command=lambda i=idx: (self._load(i), self._compute())
                      ).pack(fill="x", pady=(6, 0))
        else:
            W2kButton(inner, text="  >> 파일에서 불러오기...",
                      font=self._fonts["b"],
                      command=lambda i=idx: (self._load_from_file(i), self._compute())
                      ).pack(fill="x", pady=(6, 0))

        return outer

    # ── 동작 ──────────────────────────────────────────────────────────────────

    def _load_from_file(self, idx):
        path = filedialog.askopenfilename(
            title="이미지 파일 선택",
            filetypes=[
                ("이미지 파일", "*.jpg *.jpeg *.png *.bmp *.gif *.webp"),
                ("모든 파일",   "*.*"),
            ],
        )
        if not path:          # 취소 시 무시
            return
        self._display_image(idx, path)

    def _load(self, idx):
        self._display_image(idx, random.choice(self.all_imgs))

    def _display_image(self, idx, path):
        self.paths[idx] = path

        # 데이터셋 경로면 클래스 자동 감지, 외부 파일이면 None
        if "ants" in path:
            cls = "ants"
        elif "bees" in path:
            cls = "bees"
        else:
            cls = None
        self.cls[idx] = cls

        img = Image.open(path).convert("RGB").resize(
            (DISP_SZ, DISP_SZ), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.photos[idx] = photo

        getattr(self, f"_lbl_img{idx}").config(image=photo)

        cls_kor = {"ants": "Ants  (개미)", "bees": "Bees  (벌)", None: "알 수 없음"}
        getattr(self, f"_lbl_cls{idx}").config(text=f"클래스 : {cls_kor[cls]}")

        fname = os.path.basename(path)
        getattr(self, f"_lbl_fn{idx}").config(
            text=(fname[:34] + "...") if len(fname) > 34 else fname)

    def _compute(self):
        if None in self.paths:
            return
        self.lbl_status.config(text="유사도 계산 중 ...")
        self.update_idletasks()

        sim  = cosine_sim(self.model, self.paths[0], self.paths[1])
        pct  = (sim + 1) / 2          # [-1,1] → [0,1]
        same = sim > THRESHOLD

        self.lbl_score.config(
            text=f"{pct * 100:.1f}%",
            fg=W2K["pos"] if same else W2K["neg"])
        self.bar.set(pct, positive=same)
        self.lbl_raw.config(text=f"{sim:+.4f}")

        label = "[ SIMILAR ]  같은 종류" if same else "[ DIFFERENT ]  다른 종류"
        self.lbl_verdict.config(text=label,
                                fg=W2K["pos"] if same else W2K["neg"])

        kor = {"ants": "개미", "bees": "벌", None: "알 수 없음"}
        c0, c1 = kor[self.cls[0]], kor[self.cls[1]]
        if None in self.cls:
            verdict_ok = "---"
        else:
            correct    = (self.cls[0] == self.cls[1]) == same
            verdict_ok = "정답" if correct else "오답"
        self.lbl_status.config(
            text=f"이미지1 = {c0}    이미지2 = {c1}"
                 f"    |    cosine sim = {sim:+.4f}"
                 f"    |    판정 = {verdict_ok}")


if __name__ == "__main__":
    import tkinter.font   # noqa — font 모듈 사전 임포트
    app = App()
    app.mainloop()
