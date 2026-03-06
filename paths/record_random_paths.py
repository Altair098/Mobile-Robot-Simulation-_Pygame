# record_random_paths.py
import os, math, time
import numpy as np
import pygame

# ---------- storage ----------
SAVE_DIR = "paths"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- pygame ----------
pygame.init()
W, H = 900, 900
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Draw multiple paths: ENTER=preview  S=save  N=new  C=clear  Q=quit")
clock = pygame.time.Clock()
font = pygame.font.SysFont("arial", 18)

SCALE = 100.0  # pixels per meter
def w2s(x, y): return int(W/2 + x*SCALE), int(H/2 - y*SCALE)
def s2w(px,py): return (px - W/2)/SCALE, (H/2 - py)/SCALE

drawing = False
pts_screen = []        # current path (screen pixels)
paths_saved = 0
preview_mode = False   # True after ENTER

def draw_grid():
    screen.fill((255,255,255))
    # axes
    pygame.draw.line(screen, (230,230,230), (W//2, 0), (W//2, H), 1)
    pygame.draw.line(screen, (230,230,230), (0, H//2), (W, H//2), 1)

def draw_path(points, color=(0,0,0), step=1, radius=2):
    for i in range(0, len(points), step):
        pygame.draw.circle(screen, color, points[i], radius)

def simplify(points_px, tol=2.0):
    """Thin points to reduce file size; keep point if far enough from last kept."""
    if not points_px: return points_px
    out = [points_px[0]]
    lx, ly = points_px[0]
    for (x,y) in points_px[1:]:
        if (x-lx)**2 + (y-ly)**2 >= tol*tol:
            out.append((x,y)); lx,ly = x,y
    if out[-1] != points_px[-1]: out.append(points_px[-1])
    return out

running = True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_q:
                running = False
            elif e.key == pygame.K_RETURN:
                # preview / freeze
                preview_mode = True
            elif e.key == pygame.K_s and preview_mode and len(pts_screen) >= 5:
                # save current path (world coords)
                thin = simplify(pts_screen, tol=2.0)
                waypoints = np.array([s2w(px,py) for (px,py) in thin], dtype=float)
                ts = time.strftime("%Y%m%d_%H%M%S")
                fname = os.path.join(SAVE_DIR, f"freehand_{ts}_{paths_saved+1:02d}.npy")
                np.save(fname, waypoints)
                paths_saved += 1
                preview_mode = False
                pts_screen = []
                print(f"[saved] {fname}  {waypoints.shape}")
            elif e.key == pygame.K_n:
                # start a new path (clear current)
                preview_mode = False
                pts_screen = []
            elif e.key == pygame.K_c:
                # clear everything on screen, keep counter
                preview_mode = False
                pts_screen = []

        elif e.type == pygame.MOUSEBUTTONDOWN and not preview_mode:
            if e.button == 1:
                drawing = True
                pts_screen.append(e.pos)
        elif e.type == pygame.MOUSEMOTION and drawing and not preview_mode:
            pts_screen.append(e.pos)
        elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
            drawing = False

    draw_grid()

    # current sketch
    if pts_screen:
        draw_path(pts_screen, color=(0,120,255), step=1, radius=2)

    # preview overlay (connect with dotted green)
    if preview_mode and len(pts_screen) >= 2:
        for i in range(len(pts_screen)-1):
            if i % 2 == 0:
                pygame.draw.line(screen, (0,180,0), pts_screen[i], pts_screen[i+1], 2)

    # HUD
    hud = [
        "Draw with left mouse.",
        "ENTER: preview/freeze   S: save current   N: new path   C: clear   Q: quit",
        f"saved paths this session: {paths_saved}",
        "Tip: aim for ~300–800 points; we thin automatically before saving.",
    ]
    for i, t in enumerate(hud):
        screen.blit(font.render(t, True, (40,40,40)), (10, 10 + 22*i))

    pygame.display.flip()
    clock.tick(120)

pygame.quit()
