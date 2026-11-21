"""Bake the static map into a 2 m navigation grid (run once, offline; the map never changes).

    python3 tools/build_nav.py <path to arena dir> nav.bin

Per cell (row-major, j = y):
  blocked  uint8   1 = a mech (radius 2.5 m) can't stand here (building, cliff, map edge)
  ground   int16   terrain height, cm
  wall     int16   top of the tallest solid over the cell, cm (for line-of-sight / cover); -32768 = none
"""
import array
import json
import math
import struct
import sys
from pathlib import Path

CELL = 2.0
INFLATE = 3.0      # mech radius 2.5 + margin
STEP = 1.2
HEAD = 9.0
EDGE = 32.0        # keep away from the map edge (arena_margin is 30)


def main():
    arena, out = Path(sys.argv[1]), Path(sys.argv[2])
    w = json.loads((arena / "v2/world.json").read_text())
    t = w["terrain"]
    tnx, tny, tminx, tminy, tstep = t["nx"], t["ny"], t["minX"], t["minY"], t["step"]
    ter = array.array("h")
    ter.frombytes((arena / "v2/terrain.i16").read_bytes())
    if sys.byteorder != "little":
        ter.byteswap()

    def ground(x, y):
        fx = min(max((x - tminx) / tstep, 0), tnx - 1.001)
        fy = min(max((y - tminy) / tstep, 0), tny - 1.001)
        i, j = int(fx), int(fy)
        ax, ay = fx - i, fy - j
        k = j * tnx + i
        h00, h10, h01, h11 = ter[k], ter[k + 1], ter[k + tnx], ter[k + tnx + 1]
        return ((h00 * (1 - ax) + h10 * ax) * (1 - ay) + (h01 * (1 - ax) + h11 * ax) * ay) / 100.0

    b = w["bounds"]
    minx, miny = b["minX"], b["minY"]
    nx = int((b["maxX"] - minx) / CELL) + 1
    ny = int((b["maxY"] - miny) / CELL) + 1
    N = nx * ny
    gnd = [0.0] * N
    for j in range(ny):
        for i in range(nx):
            gnd[j * nx + i] = ground(minx + (i + 0.5) * CELL, miny + (j + 0.5) * CELL)

    blocked = bytearray(N)
    wall = [-327.68] * N

    # map edge
    for j in range(ny):
        for i in range(nx):
            x, y = minx + (i + 0.5) * CELL, miny + (j + 0.5) * CELL
            if x < b["minX"] + EDGE or x > b["maxX"] - EDGE or y < b["minY"] + EDGE or y > b["maxY"] - EDGE:
                blocked[j * nx + i] = 1

    # cliffs: height change across a cell steeper than ~50 degrees
    lim = CELL * math.tan(math.radians(50))
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            k = j * nx + i
            g = gnd[k]
            if (abs(gnd[k + 1] - g) > lim or abs(gnd[k - 1] - g) > lim
                    or abs(gnd[k + nx] - g) > lim or abs(gnd[k - nx] - g) > lim):
                blocked[k] = 1

    def poly(flat):
        return [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]

    def inside(px, py, pts):
        c = False
        n = len(pts)
        for a in range(n):
            x1, y1 = pts[a]
            x2, y2 = pts[(a + 1) % n]
            if (y1 > py) != (y2 > py) and px < (x2 - x1) * (py - y1) / (y2 - y1) + x1:
                c = not c
        return c

    def edge_dist2(px, py, pts):
        best = 1e18
        n = len(pts)
        for a in range(n):
            x1, y1 = pts[a]
            x2, y2 = pts[(a + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            L = dx * dx + dy * dy
            u = 0.0 if L == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / L))
            ex, ey = x1 + u * dx - px, y1 + u * dy - py
            d = ex * ex + ey * ey
            if d < best:
                best = d
        return best

    r2 = INFLATE * INFLATE
    for n_p, p in enumerate(w["prisms"]):
        ring = poly(p["ring"])
        holes = [poly(h) for h in p["holes"]]
        rings = [ring] + holes
        xs, ys = [q[0] for q in ring], [q[1] for q in ring]
        i0 = max(0, int((min(xs) - INFLATE - minx) / CELL))
        i1 = min(nx - 1, int((max(xs) + INFLATE - minx) / CELL) + 1)
        j0 = max(0, int((min(ys) - INFLATE - miny) / CELL))
        j1 = min(ny - 1, int((max(ys) + INFLATE - miny) / CELL) + 1)
        base, top = p["base"], p["top"]
        for j in range(j0, j1 + 1):
            py = miny + (j + 0.5) * CELL
            for i in range(i0, i1 + 1):
                px = minx + (i + 0.5) * CELL
                k = j * nx + i
                g = gnd[k]
                solid_here = top > g + STEP and base < g + HEAD   # can't walk through or under it
                ins = inside(px, py, ring) and not any(inside(px, py, h) for h in holes)
                if ins and base < g + HEAD and top > wall[k]:
                    wall[k] = top
                if not solid_here or blocked[k]:
                    continue
                if ins or min(edge_dist2(px, py, r) for r in rings) < r2:
                    blocked[k] = 1
        if n_p % 25 == 0:
            print(f"prism {n_p}/{len(w['prisms'])}", file=sys.stderr)

    with open(out, "wb") as f:
        f.write(b"NAV1")
        f.write(struct.pack("<ffffii", minx, miny, CELL, 0.0, nx, ny))
        f.write(bytes(blocked))
        f.write(array.array("h", [max(-32768, min(32767, int(round(g * 100)))) for g in gnd]).tobytes())
        f.write(array.array("h", [max(-32768, min(32767, int(round(v * 100)))) for v in wall]).tobytes())
    free = N - sum(blocked)
    print(f"grid {nx}x{ny}, {free}/{N} free cells -> {out}")


if __name__ == "__main__":
    main()
