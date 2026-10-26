"""Draw the RockWizard icon (wizard hat with a lightning bolt) -> web/icon-*.png and rockwizard.ico.
python tools/make_icon.py"""
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
N = 2048  # drawn large, then scaled down for smooth edges


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def bezier(p0, p1, p2, steps=60):
    return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
             (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]) for t in [i / steps for i in range(steps + 1)]]


def s(*pts):  # icon designed on a 1024 grid
    return [(x * N / 1024, y * N / 1024) for x, y in pts]


def draw() -> Image.Image:
    # background: diagonal violet -> magenta gradient on a rounded tile
    grad = Image.new("RGB", (N, N))
    gp = grad.load()
    c0, c1, c2 = (52, 26, 140), (139, 72, 255), (255, 76, 168)
    for y in range(0, N, 4):
        for x in range(0, N, 4):
            t = (x + y) / (2 * N)
            col = lerp(c0, c1, t / 0.6) if t < 0.6 else lerp(c1, c2, (t - 0.6) / 0.4)
            for dy in range(4):
                for dx in range(4):
                    gp[x + dx, y + dy] = col
    mask = Image.new("L", (N, N), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, N - 1, N - 1], radius=int(N * 0.225), fill=255)
    img = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    img.paste(grad, (0, 0), mask)
    d = ImageDraw.Draw(img)

    # soft glow behind the hat
    glow = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse(s((230, 250), (800, 820)), fill=(255, 255, 255, 38))
    img = Image.alpha_composite(img, glow)
    d = ImageDraw.Draw(img)

    hat = (16, 14, 40)
    # hat cone: a curved, slightly bent cone
    left = bezier(*s((300, 700), (420, 470), (585, 150)))
    tip = bezier(*s((585, 150), (660, 120), (700, 190)))
    right = bezier(*s((700, 190), (640, 260), (730, 700)))
    d.polygon(left + tip + right, fill=hat)
    # brim
    d.ellipse(s((175, 650), (860, 800)), fill=hat)
    # band
    band = bezier(*s((318, 650), (515, 700), (722, 650))) + bezier(*s((728, 600), (515, 650), (330, 600)))
    d.polygon(band, fill=(255, 210, 63))
    # lightning bolt on the cone
    d.polygon(s((560, 270), (470, 450), (540, 450), (488, 590), (640, 395), (565, 395), (620, 270)), fill=(255, 210, 63))
    # sparkles
    for cx, cy, r in ((250, 290, 60), (820, 330, 42), (790, 160, 26)):
        d.polygon(s((cx, cy - r), (cx + r * .28, cy - r * .28), (cx + r, cy), (cx + r * .28, cy + r * .28),
                    (cx, cy + r), (cx - r * .28, cy + r * .28), (cx - r, cy), (cx - r * .28, cy - r * .28)),
                  fill=(255, 255, 255))
    return img


def main():
    big = draw()
    web = ROOT / "web"
    for size in (512, 192, 180):
        im = big.resize((size, size), Image.LANCZOS)
        if size == 180:  # iOS home screen icons are square; iOS rounds them itself
            bg = Image.new("RGBA", (size, size), (52, 26, 140, 255))
            bg.alpha_composite(im)
            im = bg
        im.save(web / f"icon-{size}.png")
    big.resize((256, 256), Image.LANCZOS).save(ROOT / "rockwizard.ico",
                                               sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("icons written")


if __name__ == "__main__":
    main()
