# generic_trajectory.py
import math
import numpy as np

def _angdiff(a,b): 
    d = (a-b+math.pi)%(2*math.pi)-math.pi
    return d
def _angle_wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

class GenericTrajectory:
    """
    Path-agnostic trajectory with arc-length parameterization.
    API: sample(s), total_length(), nearest_s(), nearest_s_local(), tangent_normal(), curvature()
    """
    def __init__(self, waypoints: np.ndarray, closed: bool = True, speed: float = 0.10):
        assert waypoints.ndim == 2 and waypoints.shape[1] == 2, "waypoints shape (N,2)"
        self.closed = bool(closed)
        self.speed = float(speed)

        P = np.asarray(waypoints, dtype=float)
        # remove duplicate consecutive points
        keep = [0]
        for i in range(1, len(P)):
            if np.linalg.norm(P[i] - P[keep[-1]]) > 1e-9:
                keep.append(i)
        P = P[keep]
        if self.closed:
            if np.linalg.norm(P[0] - P[-1]) > 1e-9:
                P = np.vstack([P, P[0]])
        self.P = P  # (M+1, 2) polyline vertices along the path

        # segments, lengths, cumulative arc-length
        seg = np.diff(P, axis=0)                 # (M,2)
        self.seg = seg
        self.seg_len = np.linalg.norm(seg, axis=1)  # (M,)
        self.seg_len[self.seg_len < 1e-12] = 1e-12  # guard
        self.S = np.concatenate([[0.0], np.cumsum(self.seg_len)])  # (M+1,)
        self.L = float(self.S[-1]) if self.S[-1] > 0 else 1.0

        # segment tangents & angles
        self.t_hat = (seg.T / self.seg_len).T           # (M,2)
        self.theta_seg = np.arctan2(self.t_hat[:,1], self.t_hat[:,0])  # (M,)

    @classmethod
    def from_waypoints(cls, waypoints, closed=True, speed=0.10):
        return cls(np.asarray(waypoints, dtype=float), closed=closed, speed=speed)

    def total_length(self) -> float:
        return self.L
    
    def nearest_s_pruned(self, x, y, theta, s_prev, max_jump=0.08, cone_deg=70.0):
        """Local projection with heading gating + no backward retreat."""
        # coarse window around previous s
        N = len(self.P)
        i0 = int(max(0, min((len(self.S)-1)*s_prev, len(self.S)-2)))
        W  = max(2, int(max_jump*(len(self.S)-1)))
        i_lo = max(0, i0-W)
        i_hi = min(len(self.S)-2, i0+W)

        p  = np.array([x,y], float)
        best_d2, best_slen = 1e18, self.S[i0]

        for i in range(i_lo, i_hi+1):
            a = self.P[i]; b = self.P[i+1]; ab = b-a
            L2 = float(np.dot(ab,ab))+1e-12
            t  = float(np.dot(p-a,ab)/L2); t = 0.0 if t<0.0 else (1.0 if t>1.0 else t)
            proj = a + t*ab
            # discard candidates whose tangent faces away from robot heading
            t_hat = ab/np.linalg.norm(ab)
            th_seg = math.atan2(t_hat[1], t_hat[0])
            if abs(_angdiff(th_seg, theta)) > math.radians(cone_deg):
                continue
            d2 = float(np.dot(p-proj, p-proj))
            if d2 < best_d2:
                best_d2 = d2
                best_slen = float(self.S[i] + t*np.linalg.norm(ab))

        # fallback: if all discarded, use plain local nearest
        if best_d2 == 1e18:
            return self.nearest_s(x,y)

        s = best_slen / (self.L if self.L>0 else 1.0)
        # **monotone**: never retreat on open paths
        if not self.closed:
            s = max(s_prev, s)
        return float(np.clip(s, 0.0, 1.0))

    def _locate(self, s: float):
        # map s∈[0,1] → arc length ℓ, then segment index i and local τ∈[0,1]
        s = float(min(1.0, max(0.0, s)))
        ell = s * self.L #turn s into real distance
        i = int(np.searchsorted(self.S, ell, side='right') - 1)
        i = max(0, min(i, len(self.seg_len) - 1))
        ell0, ell1 = self.S[i], self.S[i+1]
        tau = 0.0 if ell1 <= ell0 else (ell - ell0) / (ell1 - ell0) #compute a local fraction
        return i, tau # you are at tau fraction along segment i

    def _interp_tangent(self, i: int, tau: float): # smooth direction at corners
        # smooth tangent across corners by blending adjacent segment directions
        M = len(self.seg_len)
        j = (i + 1) % M if self.closed else min(i + 1, M - 1)
        v = (1.0 - tau) * self.t_hat[i] + tau * self.t_hat[j]
        nrm = np.linalg.norm(v)
        if nrm < 1e-12:
            v = self.t_hat[i]
            nrm = np.linalg.norm(v)
        v /= nrm
        return v  # unit tangent

    def sample(self, s: float):
        """
        Return (x_d, v_d, u_ff) at normalized progress s∈[0,1].
        x_d = [x, y, theta]
        v_d = [vx, vy] (tangent direction * speed)
        u_ff = [v, omega] with omega ≈ v * curvature
        """
        i, tau = self._locate(s)   #gives the tangent vector that points along the path, also belnds the direction of segment i and the next one to stop jerks
        p = (1.0 - tau) * self.P[i] + tau * self.P[i+1]
        t_hat = self._interp_tangent(i, tau) # forward direction at this point
        theta = math.atan2(t_hat[1], t_hat[0]) #convert tangent vector into angle (heading)

        # curvature estimate via central difference of segment angles over arc-length---- to know how much path bends
        M = len(self.seg_len)
        i_prev = (i - 1) % M if self.closed else max(0, i - 1)
        i_next = (i + 1) % M if self.closed else min(M - 1, i + 1)
        dth = _angle_wrap(self.theta_seg[i_next] - self.theta_seg[i_prev]) #change in angle from the previous segment to the next one (wrapped between −π and π).
        ds_len = 0.5 * (self.seg_len[i_prev] + self.seg_len[i])
        kappa = 0.0 if ds_len < 1e-9 else (dth / ds_len) #curvature: large if the direction changes a lot over a short distance (tight curve), small if the path is nearly straight.

        v = self.speed
        omega = v * kappa
        x_d = np.array([p[0], p[1], theta], dtype=float)
        v_d = v * t_hat.copy()
        u_ff = np.array([v, omega], dtype=float)
        return x_d, v_d, u_ff

    def tangent_normal(self, s: float): #calculate the sideways error (distance off the path)
        i, tau = self._locate(s)
        t_hat = self._interp_tangent(i, tau)
        n_hat = np.array([-t_hat[1], t_hat[0]])
        return t_hat, n_hat

    def curvature(self, s: float) -> float:
        i, tau = self._locate(s)
        M = len(self.seg_len)
        i_prev = (i - 1) % M if self.closed else max(0, i - 1)
        i_next = (i + 1) % M if self.closed else min(M - 1, i + 1)
        dth = _angle_wrap(self.theta_seg[i_next] - self.theta_seg[i_prev])
        ds_len = 0.5 * (self.seg_len[i_prev] + self.seg_len[i])
        return 0.0 if ds_len < 1e-9 else (dth / ds_len)

    def nearest_s(self, px: float, py: float) -> float:   # Project a point back onto the path
        """Project point onto polyline, return s∈[0,1]."""
        p = np.array([px, py], dtype=float)
        best_d2, best_len = 1e18, 0.0
        for i in range(len(self.seg_len)):  # segment index
            a = self.P[i]; b = self.P[i+1]; ab = b - a
            L2 = float(np.dot(ab, ab)) + 1e-12
            t = float(np.dot(p - a, ab) / L2)
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
            proj = a + t * ab
            d2 = float(np.dot(p - proj, p - proj))
            if d2 < best_d2:
                best_d2 = d2
                best_len = self.S[i] + t * self.seg_len[i]
        return float(best_len / self.L if self.L > 0 else 0.0)

    def get_desired_state(self, t: float):
        """Map time→s using constant speed; wrap if closed, clamp if open."""
        s = (self.speed * max(0.0, t)) / self.L
        if self.closed:
            s = s % 1.0
        else:
            s = min(1.0, s)
        x_d, v_d, u_ff = self.sample(s)
        v_d_star = self.speed
        omega_d = u_ff[1]
        return x_d[0], x_d[1], x_d[2], v_d_star, omega_d

    def nearest_s_local(self, x, y, s_prev, max_jump=0.08):
        """
        Project (x,y) onto the path but only within a window around s_prev,
        so phase can't jump across nearby lobes/self-intersections.
        Returns s in [0,1]; allows tiny backtracking (0.01) to stay smooth.
        """
        if self.P.shape[0] < 2 or self.L <= 0:
            return float(s_prev)

        p = np.array([x, y], dtype=float)
        M = len(self.seg_len)              # number of segments
        i0 = int(np.clip(s_prev * M, 0, M - 1))
        W  = max(2, int(max_jump * M))     # window size in segments

        i_lo = max(0,     i0 - W)
        i_hi = min(M - 1, i0 + W)

        best_slen, best_d2 = self.S[i0], 1e18

        for i in range(i_lo, i_hi + 1):
            a = self.P[i]; b = self.P[i+1]; ab = b - a
            L2 = float(np.dot(ab, ab)) + 1e-12
            t  = float(np.dot(p - a, ab) / L2)
            if t < 0.0: t = 0.0
            elif t > 1.0: t = 1.0
            proj = a + t * ab
            d2   = float(np.dot(p - proj, p - proj))
            if d2 < best_d2:
                best_d2 = d2
                best_slen = float(self.S[i] + t * self.seg_len[i])

        s = best_slen / self.L
        # prevent big backward jumps; allow a tiny retreat
        s = max(s_prev - 0.01, s)
        s = float(np.clip(s, 0.0, 1.0))
        return s

    def curvature_max(self, s: float, ds: float = 0.06) -> float:
        """Max |kappa| on [s, s+ds] (clamped to [0,1])."""
        s0 = float(max(0.0, min(1.0, s)))
        s1 = float(max(0.0, min(1.0, s0 + ds)))
        if s1 <= s0 + 1e-6:
            return abs(self.curvature(s0))
        # sample 9 points in window
        ss = np.linspace(s0, s1, 9)
        return float(max(abs(self.curvature(si)) for si in ss))

    def curvature_max(self, s: float, ds: float) -> float:
        """Max |curvature| in [s-ds, s+ds]."""
        s0 = max(0.0, s - ds); s1 = min(1.0, s + ds)
        K = []
        for u in np.linspace(s0, s1, 9):
            K.append(abs(self.curvature(u)))
        return float(max(K) if K else 0.0)

    def nearest_s_pruned(self, x: float, y: float, theta: float,
                        s_prev: float, max_jump: float = 0.08,
                        cone_deg: float = 70.0) -> float:
        """
        Local projection around s_prev, but DISCARDS candidates whose tangent
        disagrees with robot heading. Prevents backward snaps.
        """
        i0 = int(np.clip(s_prev * (len(self.seg_len)), 0, len(self.seg_len)-1))
        W  = max(2, int(max_jump * len(self.seg_len)))
        i_lo = max(0, i0 - W); i_hi = min(len(self.seg_len)-1, i0 + W)

        p  = np.array([x, y], dtype=float)
        best_d2 = 1e18; best_slen = self.S[i0]
        cos_th  = math.cos(theta); sin_th = math.sin(theta)
        cos_cone = math.cos(math.radians(cone_deg))

        for i in range(i_lo, i_hi+1):
            a = self.P[i]; b = self.P[i+1]; ab = b - a
            L2 = float(np.dot(ab, ab)) + 1e-12
            t  = float(np.dot(p - a, ab) / L2); t = 0.0 if t < 0 else (1.0 if t > 1 else t)
            proj = a + t*ab
            d2   = float(np.dot(p - proj, p - proj))
            # tangent alignment gate
            t_hat = ab / (np.linalg.norm(ab) + 1e-12)
            c_t   = t_hat[0]*cos_th + t_hat[1]*sin_th
            if c_t < cos_cone:   # bad alignment → penalize heavily
                d2 *= 10.0
            if d2 < best_d2:
                best_d2   = d2
                best_slen = float(self.S[i] + t*np.linalg.norm(ab))

        s = best_slen / (self.L if self.L > 0 else 1.0)
        # allow tiny retreat only
        #s = max(s_prev - 0.01, s)
        s = max(s_prev, s)
        return float(np.clip(s, 0.0, 1.0))
