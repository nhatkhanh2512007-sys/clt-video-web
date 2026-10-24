from pathlib import Path

import imageio_ffmpeg
import matplotlib
import numpy as np

matplotlib.use("Agg")
matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

import matplotlib.pyplot as plt
from matplotlib import animation

# ---------- Cấu hình ----------
SAMPLE_SIZES = [1, 5, 30, 100]   # các kích thước mẫu n
N_REPS = 5000                    # số trung bình mẫu cho mỗi n
FPS = 24
GROW_FRAMES = 60                 # số khung hình histogram "lớn dần"
HOLD_FRAMES = 36                 # số khung hình giữ nguyên kết quả
MU, SIGMA = 1.0, 1.0             # kỳ vọng và độ lệch chuẩn của Exp(1)

OUT = Path(__file__).parent / "static" / "video" / "clt_simulation.mp4"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ---------- Sinh dữ liệu ----------
rng = np.random.default_rng(2026)
Z = {}
for n in SAMPLE_SIZES:
    samples = rng.exponential(scale=1.0, size=(N_REPS, n))
    xbar = samples.mean(axis=1)
    Z[n] = (xbar - MU) / (SIGMA / np.sqrt(n))   # chuẩn hóa Z_n

bins = np.linspace(-4, 4, 41)
centers = (bins[:-1] + bins[1:]) / 2
width = bins[1] - bins[0]

x = np.linspace(-4, 4, 400)
pdf = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

# Danh sách khung hình: (chỉ số n, số trung bình đã vẽ)
frames = []
for s in range(len(SAMPLE_SIZES)):
    for i in range(GROW_FRAMES):
        k = int(10 + (N_REPS - 10) * (i / (GROW_FRAMES - 1)) ** 2)
        frames.append((s, k))
    frames += [(s, N_REPS)] * HOLD_FRAMES

# ---------- Vẽ hình ----------
fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
fig.suptitle("Định lí Giới hạn Trung tâm", fontsize=22, fontweight="bold")
ax.set_xlim(-4, 4)
ax.set_ylim(0, 1.1)
ax.set_xlabel(r"$Z_n = \frac{\overline{X}_n - \mu}{\sigma / \sqrt{n}}$", fontsize=16)
ax.set_ylabel("Mật độ", fontsize=14)

bars = ax.bar(
    centers, np.zeros_like(centers), width=width * 0.95,
    color="#0b79d0", alpha=0.85, label="Histogram của các trung bình mẫu",
)
ax.plot(x, pdf, color="#d62728", lw=2.5, label="Chuẩn tắc N(0, 1)")
ax.legend(loc="upper right", fontsize=13)

n_text = ax.text(0.02, 0.93, "", transform=ax.transAxes,
                 fontsize=26, fontweight="bold", color="#083b70")
k_text = ax.text(0.02, 0.85, "", transform=ax.transAxes, fontsize=14)
fig.text(0.5, 0.02,
         "Phân phối gốc: mũ Exp(1), lệch phải, không phải phân phối chuẩn",
         ha="center", fontsize=12, color="#444444")
fig.tight_layout(rect=(0, 0.04, 1, 0.94))


def update(i):
    s, k = frames[i]
    n = SAMPLE_SIZES[s]
    counts, _ = np.histogram(Z[n][:k], bins=bins)
    density = counts / (k * width)
    for rect, h in zip(bars, density):
        rect.set_height(h)
    n_text.set_text(f"n = {n}")
    k_text.set_text(f"Số trung bình mẫu đã vẽ: {k:,}")
    return list(bars) + [n_text, k_text]


anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)

writer = animation.FFMpegWriter(
    fps=FPS,
    codec="libx264",
    extra_args=["-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart"],
)


def progress(current, total):
    if current % 50 == 0 or current == total - 1:
        print(f"Đang tạo video: {current + 1}/{total} khung hình")


anim.save(str(OUT), writer=writer, dpi=100, progress_callback=progress)
print("Xong! Video nằm ở:", OUT)