"""Navigation on the baked 2 m grid (nav.bin, made by tools/build_nav.py).

- free(k): can a mech stand in cell k
- path planning: A* that runs in small time slices so a tick never goes over budget
- line_free: straight walk between two points crosses no blocked cell
- los: line of sight against building tops (trees ignored: they don't stop shots)
"""
import array
import heapq
import math
import struct
import time
from pathlib import Path

SQ2 = math.sqrt(2.0)


class Nav:
    def __init__(self, path):
        d = Path(path).read_bytes()
        assert d[:4] == b"NAV1", "bad nav file"
        self.minx, self.miny, self.cell, _, self.nx, self.ny = struct.unpack("<ffffii", d[4:28])
        n = self.nx * self.ny
        self.n = n
        self.blocked = bytearray(d[28:28 + n])
        self.ground = array.array("h")
        self.ground.frombytes(d[28 + n:28 + 3 * n])
        self.wall = array.array("h")
        self.wall.frombytes(d[28 + 3 * n:28 + 5 * n])
        nx = self.nx
        # (offset, cost, the two orthogonal cells a diagonal step passes between)
        self.nbrs = [(1, 1.0, 0, 0), (-1, 1.0, 0, 0), (nx, 1.0, 0, 0), (-nx, 1.0, 0, 0),
                     (nx + 1, SQ2, 1, nx), (nx - 1, SQ2, -1, nx),
                     (-nx + 1, SQ2, 1, -nx), (-nx - 1, SQ2, -1, -nx)]
        self.g = array.array("d", [0.0]) * n
        self.stamp = array.array("i", [0]) * n
        self.parent = array.array("i", [0]) * n
        self.sid = 0

    # --- cells -------------------------------------------------------------
    def k_of(self, x, y):
        i = int((x - self.minx) / self.cell)
        j = int((y - self.miny) / self.cell)
        i = 0 if i < 0 else (self.nx - 1 if i >= self.nx else i)
        j = 0 if j < 0 else (self.ny - 1 if j >= self.ny else j)
        return j * self.nx + i

    def xy(self, k):
        j, i = divmod(k, self.nx)
        return (self.minx + (i + 0.5) * self.cell, self.miny + (j + 0.5) * self.cell)

    def free(self, k):
        return not self.blocked[k]

    def free_xy(self, x, y):
        return not self.blocked[self.k_of(x, y)]

    def ground_at(self, x, y):
        return self.ground[self.k_of(x, y)] / 100.0

    def wall_at(self, x, y):
        return self.wall[self.k_of(x, y)] / 100.0

    def nearest_free(self, k, maxr=25):
        if not self.blocked[k]:
            return k
        j0, i0 = divmod(k, self.nx)
        for r in range(1, maxr + 1):
            best, bd = None, 1e9
            for dj in range(-r, r + 1):
                for di in (-r, r) if abs(dj) != r else range(-r, r + 1):
                    i, j = i0 + di, j0 + dj
                    if 0 <= i < self.nx and 0 <= j < self.ny:
                        kk = j * self.nx + i
                        if not self.blocked[kk]:
                            d = di * di + dj * dj
                            if d < bd:
                                best, bd = kk, d
            if best is not None:
                return best
        return None

    def cover_score(self, k, r=3):
        """How many blocked-by-building cells are near k (more = better cover)."""
        j0, i0 = divmod(k, self.nx)
        s = 0
        for dj in range(-r, r + 1):
            row = (j0 + dj) * self.nx
            if row < 0 or row >= self.n:
                continue
            for di in range(-r, r + 1):
                kk = row + i0 + di
                if 0 <= kk < self.n and self.wall[kk] > -32768:
                    s += 1
        return s

    # --- lines -------------------------------------------------------------
    def line_free(self, x0, y0, x1, y1):
        d = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(d / (self.cell * 0.5)))
        for s in range(n + 1):
            u = s / n
            if self.blocked[self.k_of(x0 + (x1 - x0) * u, y0 + (y1 - y0) * u)]:
                return False
        return True

    def los(self, a, b):
        """True if no building top rises above the straight line from a to b (3D points)."""
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(d / 1.5))
        wall = self.wall
        for s in range(1, n):
            u = s / n
            k = self.k_of(a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)
            w = wall[k]
            if w > -32768 and w / 100.0 > a[2] + (b[2] - a[2]) * u:
                return False
        return True

    # --- path planning -------------------------------------------------------
    def astar(self, s, goal, max_expand=60000):
        """Generator. Yields None while working (caller checks its clock), then a list of
        cells from s to goal, or [] if unreachable / too far."""
        self.sid += 1
        sid = self.sid
        g, stamp, parent, blocked, nbrs = self.g, self.stamp, self.parent, self.blocked, self.nbrs
        nx = self.nx
        gj, gi = divmod(goal, nx)

        def h(k):
            j, i = divmod(k, nx)
            dx, dy = abs(i - gi), abs(j - gj)
            return 2.0 * (dx + dy + (SQ2 - 2) * min(dx, dy))

        stamp[s] = sid
        g[s] = 0.0
        parent[s] = -1
        heap = [(h(s), 0.0, s)]
        exp = 0
        while heap:
            _, gc, k = heapq.heappop(heap)
            if gc > g[k] + 1e-6:
                continue
            if k == goal:
                path = []
                while k != -1:
                    path.append(k)
                    k = parent[k]
                path.reverse()
                yield path
                return
            exp += 1
            if exp > max_expand:
                break
            if exp & 255 == 0:
                yield None
            for off, c, o1, o2 in nbrs:
                kk = k + off
                if blocked[kk]:
                    continue
                if o1 and (blocked[k + o1] or blocked[k + o2]):
                    continue  # no cutting corners
                ng = gc + c
                if stamp[kk] != sid or ng < g[kk] - 1e-6:
                    stamp[kk] = sid
                    g[kk] = ng
                    parent[kk] = k
                    heapq.heappush(heap, (ng + h(kk), ng, kk))
        yield []


class Planner:
    """Keeps a path to a goal and gives a steering point each tick."""

    def __init__(self, nav):
        self.nav = nav
        self.job = None
        self.job_goal = None
        self.path = None
        self.goal = None
        self.idx = 0
        self.failed_until = 0.0

    def set_goal(self, gx, gy, t):
        nav = self.nav
        gk = nav.nearest_free(nav.k_of(gx, gy))
        if gk is None:
            return
        if self.goal is not None and (self.goal == gk or self._close(self.goal, gk, 4)):
            return
        self.goal = gk
        self.path = None
        self.job = None

    def _close(self, a, b, cells):
        aj, ai = divmod(a, self.nav.nx)
        bj, bi = divmod(b, self.nav.nx)
        return abs(aj - bj) <= cells and abs(ai - bi) <= cells

    def replan(self):
        self.path = None
        self.job = None

    def step(self, x, y, deadline, t):
        """Returns a point (x, y) to walk toward, or None to walk straight at the goal."""
        nav = self.nav
        if self.goal is None:
            return None
        here = nav.k_of(x, y)
        start = nav.nearest_free(here, 6)
        if start is None:
            return nav.xy(self.goal)
        # plan
        if self.path is None:
            if t < self.failed_until:
                return nav.xy(self.goal)
            if self.job is None or self.job_goal != self.goal:
                self.job = nav.astar(start, self.goal)
                self.job_goal = self.goal
            res = None
            while time.perf_counter() < deadline:
                res = next(self.job, [])
                if res is not None:
                    break
            if res is None:
                # still searching: head straight for now if possible
                return nav.xy(self.goal)
            self.job = None
            if not res:
                self.failed_until = t + 2.0
                return nav.xy(self.goal)
            self.path = res
            self.idx = 0
        path = self.path
        # find where we are on the path (search a window ahead)
        best, bd = self.idx, 1e18
        for m in range(self.idx, min(len(path), self.idx + 40)):
            px, py = nav.xy(path[m])
            d = (px - x) ** 2 + (py - y) ** 2
            if d < bd:
                best, bd = m, d
        if bd > 12.0 ** 2:  # wandered off the path
            self.replan()
            return nav.xy(self.goal)
        self.idx = best
        # steer at the farthest path point we can walk to in a straight line
        target = path[min(best + 1, len(path) - 1)]
        for m in range(min(len(path) - 1, best + 30), best, -3):
            px, py = nav.xy(path[m])
            if nav.line_free(x, y, px, py):
                target = path[m]
                break
        return nav.xy(target)

    def arrived(self, x, y, r=3.0):
        if self.goal is None:
            return True
        gx, gy = self.nav.xy(self.goal)
        return math.hypot(gx - x, gy - y) < r
