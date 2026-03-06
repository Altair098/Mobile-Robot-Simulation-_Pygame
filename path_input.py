# path_input.py
import pygame
import numpy as np
import math

def _dist2(a, b):
    dx, dy = a[0] - b[0], a[1] - b[1]
    return dx*dx + dy*dy

def moving_average_xy(points_px, window=7):
    if len(points_px) < 3 or window <= 1:
        return points_px[:]
    w = max(3, window | 1)  # force odd
    half = w // 2
    out = []
    for i in range(len(points_px)):
        i0 = max(0, i - half)
        i1 = min(len(points_px), i + half + 1)
        xs = [p[0] for p in points_px[i0:i1]]
        ys = [p[1] for p in points_px[i0:i1]]
        out.append((sum(xs)/len(xs), sum(ys)/len(ys)))
    return out

def resample_by_arclength(points_px, N=400):
    """Uniformly re-sample a polyline to N points (in pixel space)."""
    if len(points_px) < 2:
        return points_px[:]
    P = np.asarray(points_px, dtype=float)
    seg = np.diff(P, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    S = np.concatenate([[0.0], np.cumsum(seg_len)])
    L = S[-1]
    if L < 1e-6:
        return points_px[:]
    s_targets = np.linspace(0.0, L, N)
    out = []
    j = 0
    for s in s_targets:
        while j < len(S)-2 and not (S[j] <= s <= S[j+1]):
            j += 1
        t = 0.0 if S[j+1] == S[j] else (s - S[j]) / (S[j+1] - S[j])
        p = (1.0 - t) * P[j] + t * P[j+1]
        out.append((float(p[0]), float(p[1])))
    return out

class FreehandPathRecorder:
    """
    Usage:
        rec = FreehandPathRecorder(screen, world_from_screen)
        waypoints_world = rec.run()  # returns np.ndarray (N,2) in meters
    """
    def __init__(self, screen, world_from_screen,
                 min_pix_step=3, draw_color=(0,200,0), text_color=(20,20,20)):
        self.screen = screen
        self.world_from_screen = world_from_screen
        self.min_pix_step2 = float(min_pix_step)**2
        self.draw_color = draw_color
        self.text_color = text_color
        self.font = pygame.font.SysFont("arial", 16)
        self.clear()

    def clear(self):
        self.raw_px = []      # raw points while dragging
        self.paths_px = []    # committed strokes (list of list of (px,py))
        self.drawing = False

    def _blit_instructions(self):
        lines = [
            "Draw your path with LEFT mouse button.",
            "Press ENTER to finish, R to clear, ESC to cancel.",
            f"Current points: {sum(len(p) for p in self.paths_px) + len(self.raw_px)}"
        ]
        y = 8
        for s in lines:
            surf = self.font.render(s, True, self.text_color)
            self.screen.blit(surf, (8, y))
            y += 18

    def _render_paths(self):
        # draw committed strokes
        for stroke in self.paths_px:
            for i in range(1, len(stroke)):
                pygame.draw.line(self.screen, self.draw_color, stroke[i-1], stroke[i], 2)
        # draw current stroke
        for i in range(1, len(self.raw_px)):
            pygame.draw.line(self.screen, self.draw_color, self.raw_px[i-1], self.raw_px[i], 2)

    def run(self):
        """Modal capture loop; returns waypoints in WORLD (meters) as np.ndarray (N,2) or None if cancelled."""
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return None
                    if event.key == pygame.K_r:
                        self.clear()
                    if event.key == pygame.K_RETURN:
                        pts = self._finalize_points_world()
                        if pts is not None:
                            return pts
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.drawing = True
                    pos = pygame.mouse.get_pos()
                    self.raw_px = [pos]
                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    if self.drawing:
                        if len(self.raw_px) >= 2:
                            self.paths_px.append(self.raw_px[:])
                        self.raw_px = []
                        self.drawing = False
                if event.type == pygame.MOUSEMOTION and self.drawing:
                    pos = pygame.mouse.get_pos()
                    if not self.raw_px or _dist2(pos, self.raw_px[-1]) >= self.min_pix_step2:
                        self.raw_px.append(pos)

            # draw UI
            self.screen.fill((255,255,255))
            self._render_paths()
            self._blit_instructions()
            pygame.display.flip()
            clock.tick(60)

    def _finalize_points_world(self):
        # stitch all strokes into one polyline in pixel space
        pts_px = []
        for stroke in self.paths_px:
            if len(stroke) == 0:
                continue
            if not pts_px:
                pts_px.extend(stroke)
            else:
                # connect with a line if needed
                if stroke and pts_px[-1] != stroke[0]:
                    pts_px.extend(stroke)
                else:
                    pts_px.extend(stroke[1:])
        if len(pts_px) < 8:
            return None

        # smooth + resample in PIXELS
        pts_px = moving_average_xy(pts_px, window=7)
        pts_px = resample_by_arclength(pts_px, N=400)

        # convert to WORLD meters
        pts_w = [self.world_from_screen(px, py) for (px, py) in pts_px]
        P = np.asarray(pts_w, dtype=float)

        # reject too-short paths
        seg = np.diff(P, axis=0)
        total_len = float(np.sum(np.linalg.norm(seg, axis=1)))
        if total_len < 0.5:  # meters
            return None

        return P
