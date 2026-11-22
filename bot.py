"""Survival-first mechfight bot.

Each tick (10 per second) it decides, in order:
  1. Where to stand: inside the NEXT zone circle, near buildings for cover, before the zone closes.
  2. Whether to run: low HP and under fire -> move somewhere the attacker can't see, and regenerate.
  3. What to shoot: the best visible enemy (close, weak, or the one hurting us), with lead on moving targets.
     Always face attackers (hits from behind do 1.5x damage).
  4. Free time: grab better weapons, power cubes, crates, NPC drones and supply drops nearby.
Legs (move) and torso (aim) are independent, so it can walk to cover while shooting backwards.
"""
import math
import os
import random
import time

import mechfight as mf
from nav import Nav, Planner

HERE = os.path.dirname(os.path.abspath(__file__))
# lightning: 240 dmg/s within 45 m and aims itself -> best for aggressive close fights
WEAPON_RANK = {"lightning": 3, "laser": 2, "mortar": 1, "gun": 0}
HUNT_HP = 0.55        # hunt while above this fraction of max HP
CLOSE_IN = 26.0       # distance we try to fight at (gun hits more, lightning reaches 45 m)
GROUP_R = 50.0        # an enemy with 2+ friends within this many meters is "in a group": don't push it
RETREAT_HP = 0.5      # below this fraction of max HP (with enemies around), hide and regenerate
CUBE_GRAB = 25.0      # grab cubes this close even mid-fight
# Simulator experiments can override any of the constants above, e.g. MF_PARAMS='{"HUNT_HP": 0.7}'.
# (The environment variable is never set in real matches.)
try:
    globals().update(__import__("json").loads(os.environ.get("MF_PARAMS", "{}")))
except ValueError:
    pass
UPGRADE_RANK = ["armor", "plating", "nanites", "gun_damage", "gun_rate", "laser_charge", "laser_damage",
                "lightning_damage", "gun_mag", "gun_velocity", "servos", "mortar_reload", "sensors",
                "gyros", "reactor", "capacitor", "jets", "mortar_splash", "mortar_velocity"]
TICK_BUDGET = 0.020  # seconds of path-planning per tick (the hard limit is 0.050)


def ang_diff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def unit(dx, dy):
    d = math.hypot(dx, dy)
    return (0.0, 0.0) if d < 1e-6 else (dx / d, dy / d)


class Bot:
    name = "survivor"

    def __init__(self, hello):
        self.cfg = hello["config"]
        random.seed(hello.get("seed", 0))
        self.nav = Nav(os.path.join(HERE, "nav.bin"))
        self.plan = Planner(self.nav)
        self.last_hit_t = -99.0      # last time we took damage
        self.last_deal_t = -99.0     # last time we dealt damage
        self.attacker = None
        self.hit_bearing = None
        self.spot = None             # where we want to hold inside the zone
        self.spot_key = None
        self.hist = []               # (t, x, y) for stuck detection
        self.unstick_until = -1.0
        self.unstick_dir = (0.0, 0.0)
        self.stuck_count = 0
        self.strafe_sign = 1
        self.strafe_flip_t = 0.0
        self.cover = None
        self.cover_t = -99.0
        self.loot_ignore = {}        # id -> time until which we ignore it
        self.mode = "start"
        self.foes = {}               # enemy id -> what we last knew (pos, vel, hp, when, damage it did to us)
        self.target_id = None        # who we're focusing
        self.fight_spot = None       # cached isolation / hiding spot
        self.fight_spot_t = -99.0
        self.fight_spot_kind = None
        self.rear_hit_t = -99.0

    # ------------------------------------------------------------------ helpers
    def zone_now(self, z, t):
        """Actual zone circle at time t."""
        cx, cy = z["center"]
        r = z["radius"]
        if t >= z["shrink_start"]:
            span = max(1e-6, z["shrink_end"] - z["shrink_start"])
            u = min(1.0, (t - z["shrink_start"]) / span)
            nx_, ny_ = z["next_center"]
            cx, cy = cx + (nx_ - cx) * u, cy + (ny_ - cy) * u
            r = r + (z["next_radius"] - r) * u
        return cx, cy, r

    def safe_point(self, x, y, z):
        """A point inside the next circle, not dead center (that's where everyone fights)."""
        ncx, ncy = z["next_center"]
        rn = z["next_radius"]
        d = math.hypot(x - ncx, y - ncy)
        frac = 0.55 if rn > 60 else 0.35
        keep = min(d, frac * rn)
        ux, uy = unit(x - ncx, y - ncy)
        return ncx + ux * keep, ncy + uy * keep

    def pick_spot(self, x, y, z):
        """Free cell near the safe point with buildings around it (cover)."""
        nav = self.nav
        sx, sy = self.safe_point(x, y, z)
        ncx, ncy = z["next_center"]
        rn = z["next_radius"]
        best, bs = None, -1e9
        for r in (0, 6, 12, 18, 26):
            for a in range(0, 360, 30 if r else 360):
                px = sx + r * math.cos(math.radians(a))
                py = sy + r * math.sin(math.radians(a))
                if math.hypot(px - ncx, py - ncy) > rn * 0.85:
                    continue
                k = nav.k_of(px, py)
                if not nav.free(k):
                    continue
                s = nav.cover_score(k, 4) - 0.15 * r - 0.01 * math.hypot(px - x, py - y)
                if s > bs:
                    best, bs = (px, py), s
        return best or (sx, sy)

    def find_cover(self, x, y, threat, z, t):
        """A nearby free point the threat can't see. Cached for a second."""
        if self.cover and t - self.cover_t < 1.0:
            return self.cover
        nav = self.nav
        cx, cy, r = self.zone_now(z, t)
        tp = (threat[0], threat[1], threat[2] + 8)
        best, bs = None, -1e9
        away = math.atan2(y - threat[1], x - threat[0])
        for dist in (8, 14, 22, 32, 44):
            for da in range(-150, 181, 30):
                a = away + math.radians(da)
                px, py = x + dist * math.cos(a), y + dist * math.sin(a)
                if math.hypot(px - cx, py - cy) > r - 8 or not nav.free_xy(px, py):
                    continue
                g = nav.ground_at(px, py)
                if nav.los(tp, (px, py, g + 6)):
                    continue  # still visible
                s = -dist + 0.3 * math.hypot(px - threat[0], py - threat[1]) * 0.2 - abs(da) * 0.03
                if s > bs:
                    best, bs = (px, py), s
        self.cover, self.cover_t = best, t
        return best

    def lead(self, me_pos, muzzle, tpos, tvel, speed, height):
        """Aim [yaw, pitch] that leads a target moving at tvel with a bullet of `speed`."""
        ox, oy, oz = me_pos[0], me_pos[1], me_pos[2] + muzzle
        px, py, pz = tpos[0], tpos[1], tpos[2] + height
        vx, vy, vz = tvel
        tt = 0.0
        for _ in range(3):
            ax, ay, az = px + vx * tt, py + vy * tt, pz + vz * tt * 0.3
            tt = math.dist((ox, oy, oz), (ax, ay, az)) / speed
        ax, ay, az = px + vx * tt, py + vy * tt, pz + vz * tt * 0.3
        h = math.hypot(ax - ox, ay - oy)
        return [math.atan2(ay - oy, ax - ox), math.atan2(az - oz, max(h, 0.1))], h

    def mortar_aim(self, me_pos, tube, tpos, tvel, v, g):
        """Low-arc launch for the mortar, leading by flight time. None if out of range."""
        ox, oy, oz = me_pos[0], me_pos[1], me_pos[2] + tube
        tt = 0.0
        sol = None
        for _ in range(3):
            ax, ay, az = tpos[0] + tvel[0] * tt, tpos[1] + tvel[1] * tt, tpos[2] + 1.0
            d = math.hypot(ax - ox, ay - oy)
            dz = az - oz
            disc = v ** 4 - g * (g * d * d + 2 * dz * v * v)
            if disc < 0 or d < 1:
                return None
            th = math.atan2(v * v - math.sqrt(disc), g * d)
            tt = d / max(1e-3, v * math.cos(th))
            sol = [math.atan2(ay - oy, ax - ox), th]
        return sol

    def aim_ok(self, me, want, dist, radius=2.5, slack=1.0):
        tol = max(0.012, math.atan2(radius * 0.8, max(dist, 1.0))) * slack
        return (abs(ang_diff(me["aim"][0], want[0])) < tol and abs(me["aim"][1] - want[1]) < tol * 1.5)

    # ------------------------------------------------------------------ main
    def act(self, obs):
        t0 = time.perf_counter()
        deadline = t0 + TICK_BUDGET
        t = obs["t"]
        me = obs["self"]
        x, y, zf = me["pos"]
        st = me["stats"]
        z = obs["zone"]
        hp_frac = me["hp"] / max(1.0, me["max_hp"])
        act = {}

        # ---- events
        for ev in obs.get("events", []):
            et = ev.get("type")
            if et == "damaged":
                self.last_hit_t = t
                by = ev.get("by")
                if by is not None:
                    self.attacker = by
                    if by < 100:
                        f = self.foes.setdefault(by, {"pos": None, "t": -99.0, "dmg": 0.0, "hit_t": -99.0})
                        f["dmg"] = f.get("dmg", 0.0) * 0.5 ** max(0.0, t - f.get("hit_t", t)) + ev.get("amount", 0)
                        f["hit_t"] = t
                if ev.get("bearing") is not None:
                    self.hit_bearing = ev["bearing"]
                if ev.get("rear"):
                    self.rear_hit_t = t
            elif et == "hit":
                self.last_deal_t = t
            elif et == "take_failed":
                self.loot_ignore[ev.get("id")] = t + 10

        # ---- upgrade offer
        off = obs.get("offer")
        if off:
            ranks = [UPGRADE_RANK.index(c["id"]) if c["id"] in UPGRADE_RANK else 99 for c in off["choices"]]
            act["pick"] = ranks.index(min(ranks))

        # ---- what's around
        enemies = [m for m in obs.get("mechs", []) if m["kind"] == "mech"]
        npcs = [m for m in obs.get("mechs", []) if m["kind"] == "npc"]
        for m in enemies:
            m["_d"] = math.dist(m["pos"], me["pos"])
        for m in npcs:
            m["_d"] = math.dist(m["pos"], me["pos"])
        since_hit = t - self.last_hit_t

        # ---- zone: where we should be
        zcx, zcy, zr = self.zone_now(z, t)
        ncx, ncy = z["next_center"]
        rn = z["next_radius"]
        key = (round(ncx), round(ncy), round(rn))
        if key != self.spot_key:
            self.spot_key = key
            self.spot = self.pick_spot(x, y, z)
        d_next = math.hypot(x - ncx, y - ncy)
        time_left = max(0.0, z["shrink_end"] - t)
        dist_to_safe = max(0.0, d_next - rn * 0.85)
        urgent = (not me["in_zone"]) or (dist_to_safe > 0 and dist_to_safe > 0.6 * st["walk_speed"] * time_left)
        # we are "home" if already well inside the next circle
        home_ok = d_next < rn * 0.8

        # ---- remember enemies (obs keeps tracking them 5 s after they leave view; we keep 12 s)
        for m in enemies:
            f = self.foes.setdefault(m["id"], {"dmg": 0.0, "hit_t": -99.0})
            f.update(pos=m["pos"], vel=m.get("vel", [0, 0, 0]), hp=m.get("hp", 1400),
                     max_hp=m.get("max_hp", 1400), t=t)
        for k in [k for k, f in self.foes.items() if t - f.get("t", -99) > 12 and t - f.get("hit_t", -99) > 12]:
            del self.foes[k]
        active = [(k, f) for k, f in self.foes.items()
                  if f.get("pos") and t - f["t"] < 8 and math.dist(f["pos"][:2], (x, y)) < 160]
        # which of them can probably see (and shoot) us right now
        body = (x, y, zf + 4.5)
        exposed = []
        for k, f in sorted(active, key=lambda kf: math.dist(kf[1]["pos"][:2], (x, y)))[:6]:
            p = f["pos"]
            if self.nav.los((p[0], p[1], p[2] + 8), body):
                exposed.append(k)
        n_exp = len(exposed)

        # ---- target: focus whoever will die fastest, and stick with them
        target = self.choose_target(enemies, me, x, y, zf, t)
        npc_target = None
        if not enemies and npcs and hp_frac > 0.5:
            close = [n for n in npcs if n["_d"] < 90]
            if close:
                npc_target = min(close, key=lambda n: n["_d"])

        # ---- decide where to go
        goal = None
        mode = "hold"
        boost = False
        finishing = target is not None and target.get("hp", 1400) < 160 and target["_d"] < 120
        hurt = hp_frac < RETREAT_HP or (hp_frac < RETREAT_HP + 0.2 and n_exp >= 2) or \
            (hp_frac < RETREAT_HP + 0.35 and n_exp >= 3)
        # power cubes lying right next to us: grab them even mid-fight (+4% HP and damage each)
        cube_near = None
        if not urgent and n_exp <= 1:
            cs = [c for c in obs.get("cubes", []) if math.dist(c["pos"][:2], (x, y)) < CUBE_GRAB
                  and abs(c["pos"][2] - zf) < 4]
            if cs:
                cube_near = min(cs, key=lambda c: math.dist(c["pos"][:2], (x, y)))["pos"][:2]
        npc_threat = min(npcs, key=lambda n: n["_d"]) if npcs and since_hit < 2 else None
        if urgent:
            mode = "zone"
            goal = self.spot
            boost = not me["in_zone"] or me["energy"] > 40
        elif active and hurt and not finishing:
            # get out of everyone's sight and regenerate
            mode = "retreat"
            goal = self.fight_pos(x, y, zf, z, t, active, None, "hide") or self.spot
            boost = me["energy"] > 20
        elif cube_near is not None and not (active and hurt):
            mode = "cube"
            goal = cube_near
        elif n_exp >= 2:
            # outnumbered: move so that only our target can see us
            mode = "isolate"
            goal = self.fight_pos(x, y, zf, z, t, active, self.target_id, "isolate") or self.spot
        elif npc_threat is not None and hp_frac < 0.4:
            mode = "retreat"
            goal = self.find_cover(x, y, npc_threat["pos"], z, t) or self.spot
            boost = me["energy"] > 20
        elif hp_frac > HUNT_HP and (me["weapons"][1] is not None or t > 45 or enemies) and \
                (hg := self.hunt_goal(enemies, target, active, obs, x, y, z, t)) is not None:
            # go get them: an isolated enemy we can see, remember, or hear
            mode = "hunt"
            goal = hg if hg != "here" else None
            boost = target is None and me["energy"] > 60
        else:
            goal = None
            if n_exp == 0 and (not enemies or hp_frac > 0.5):
                goal = self.loot_goal(obs, me, x, y, zf, z, t, enemies)
            if goal is not None:
                mode = "loot"
            elif not home_ok or math.hypot(self.spot[0] - x, self.spot[1] - y) > 6:
                mode = "rotate" if not enemies else "fight"
                goal = self.spot
            else:
                mode = "fight" if enemies else "hold"
        self.mode = mode
        # route around enemies we know about (re-plan every 2 s while they're around)
        self.plan.avoid = [(f["pos"][0], f["pos"][1], 50.0, 5.0) for k, f in active if k != self.target_id or mode == "retreat"]
        if active and t - self.plan.planned_t > 2.0:
            self.plan.replan()

        # ---- legs
        mv = (0.0, 0.0)
        if goal is not None:
            self.plan.set_goal(goal[0], goal[1], t)
            steer = self.plan.step(x, y, deadline, t)
            if steer is not None and math.hypot(goal[0] - x, goal[1] - y) > 2.5:
                mv = unit(steer[0] - x, steer[1] - y)
        # strafe while fighting so we're harder to hit
        if t > self.strafe_flip_t:
            self.strafe_sign = -self.strafe_sign
            self.strafe_flip_t = t + random.uniform(0.8, 1.8)
        if target is not None and mode in ("fight", "hold", "rotate", "hunt", "cube"):
            tx, ty = unit(target["pos"][0] - x, target["pos"][1] - y)
            px_, py_ = -ty * self.strafe_sign, tx * self.strafe_sign
            w = 0.7 if mv == (0.0, 0.0) else 0.45
            if not self.nav.free_xy(x + px_ * 4, y + py_ * 4):
                self.strafe_sign = -self.strafe_sign
                px_, py_ = -px_, -py_
            mv = unit(mv[0] + px_ * w, mv[1] + py_ * w)
            # don't let them get on top of us
            if target["_d"] < 15:
                mv = unit(mv[0] - tx * 0.6, mv[1] - ty * 0.6)
        # dodge lasers aimed at us and mortar shells landing near us
        dodge = self.dodge(obs, me, x, y, zf)
        if dodge is not None:
            mv = unit(mv[0] * 0.3 + dodge[0], mv[1] * 0.3 + dodge[1])
            boost = boost or me["energy"] > 15

        # ---- stuck? jump/jet and sidestep, then replan
        jump = False
        self.hist.append((t, x, y))
        while self.hist and t - self.hist[0][0] > 1.2:
            self.hist.pop(0)
        if t < self.unstick_until:
            mv = self.unstick_dir
            jump = me["grounded"]
            boost = True
        elif mv != (0.0, 0.0) and len(self.hist) > 8:
            ot, ox_, oy_ = self.hist[0]
            if t - ot > 1.0 and math.hypot(x - ox_, y - oy_) < 1.5:
                self.stuck_count += 1
                a = math.atan2(mv[1], mv[0]) + random.choice((-1, 1)) * random.uniform(0.6, 1.6)
                self.unstick_dir = (math.cos(a), math.sin(a)) if self.stuck_count % 3 else mv
                self.unstick_until = t + 0.8
                self.hist.clear()
                if self.stuck_count % 2 == 0:
                    self.plan.replan()
            elif math.hypot(x - ox_, y - oy_) > 4:
                self.stuck_count = 0

        # ---- torso: aim and fire
        aim, fire, extra = self.weapons(obs, me, x, y, zf, target, npc_target, mode, t)
        act.update(extra)
        # shot in the back by someone we can't see: turn to face them (rear hits do 1.5x)
        if (t - self.rear_hit_t < 0.8 and self.attacker is not None and self.attacker < 100
                and not any(e["id"] == self.attacker for e in enemies) and self.hit_bearing is not None):
            aim = [self.hit_bearing, 0.0]
        if aim is None:
            aim = self.look(obs, me, mv, t)

        act.update(move=[round(mv[0], 3), round(mv[1], 3)], boost=bool(boost), jump=jump,
                   aim=[round(aim[0], 4), round(aim[1], 4)], fire=fire)
        mf.draw.label(f"{mode} hp{me['hp']:.0f} e{len(enemies)} x{n_exp}")
        return act

    # ------------------------------------------------------------------ fights
    def choose_target(self, enemies, me, x, y, zf, t):
        """Lowest 'shots needed to kill' wins: weak and close enemies first. Sticky, so we finish kills."""
        if not enemies:
            self.target_id = None
            return None
        muzzle = (x, y, zf + me["stats"].get("muzzle_height", 7))
        best, bs = None, 1e18
        for m in enemies:
            d = m["_d"]
            hp = m.get("hp", 1400)
            p = m["pos"]
            hit = max(0.15, 1.0 - d / 220.0)            # rough chance a gun shot lands at this range
            s = hp / hit
            if not self.nav.los(muzzle, (p[0], p[1], p[2] + 4.5)):
                s *= 3.0                                 # can't shoot it from here
            if hp < 200:
                s *= 0.4                                 # nearly dead: finish it
            if m["id"] == self.target_id:
                s *= 0.6                                 # stick with our target
            f = self.foes.get(m["id"])
            if f and t - f.get("hit_t", -99) < 2:
                s *= 0.6                                 # it's hurting us right now
            if s < bs:
                best, bs = m, s
        self.target_id = best["id"]
        return best

    def fight_pos(self, x, y, zf, z, t, active, want_id, kind):
        """Pick a nearby spot by who can see it.
        kind 'hide': nobody can see it.  kind 'isolate': only enemy `want_id` can see it.
        Also prefers being far from enemies and close to where we are. Cached ~0.8 s."""
        if self.fight_spot and self.fight_spot_kind == kind and t - self.fight_spot_t < 0.8:
            return self.fight_spot
        nav = self.nav
        cx, cy, r = self.zone_now(z, t)
        ncx, ncy = z["next_center"]
        rn = z["next_radius"]
        zone_pull = z["shrink_start"] - t < 20 or math.hypot(x - ncx, y - ncy) > rn
        foes = sorted(active, key=lambda kf: math.dist(kf[1]["pos"][:2], (x, y)))[:5]
        eyes = [(k, (f["pos"][0], f["pos"][1], f["pos"][2] + 8)) for k, f in foes]
        cands = [(x, y, 0.0)]
        for dist in (7.0, 14.0, 24.0, 36.0):
            off = (dist * 7) % 36
            for a in range(0, 360, 36):
                ang = math.radians(a + off)
                cands.append((x + dist * math.cos(ang), y + dist * math.sin(ang), dist))
        best, bs = None, -1e18
        for px, py, dist in cands:
            if math.hypot(px - cx, py - cy) > r - 8:
                continue
            if dist and not nav.free_xy(px, py):
                continue
            body = (px, py, nav.ground_at(px, py) + 5)
            seen_by = 0
            sees_target = False
            crowd = 0.0
            for k, e in eyes:
                vis = nav.los(e, body)
                if k == want_id:
                    sees_target = vis
                elif vis:
                    seen_by += 1
                crowd += max(0.0, 45.0 - math.dist(e[:2], (px, py))) / 15.0
            s = -10.0 * seen_by - crowd - 0.12 * dist
            if kind == "isolate" and sees_target:
                s += 6.0
            if dist and not nav.line_free(x, y, px, py):
                s -= 3.0                                  # have to walk around something to get there
            if zone_pull:
                s -= 0.1 * max(0.0, math.hypot(px - ncx, py - ncy) - rn * 0.8)
            if s > bs:
                best, bs = (px, py), s
        self.fight_spot, self.fight_spot_t, self.fight_spot_kind = best, t, kind
        return best

    def hunt_goal(self, enemies, target, active, obs, x, y, z, t):
        """Where to go to start the next fight. 'here' = already in range, fight in place. None = no hunt."""
        cx, cy, r = self.zone_now(z, t)
        ncx, ncy = z["next_center"]
        rn = z["next_radius"]
        late = z["shrink_start"] - t < 15

        def ok(p):
            if math.hypot(p[0] - cx, p[1] - cy) > r - 15:
                return False
            if late and math.hypot(p[0] - ncx, p[1] - ncy) > rn * 0.9:
                return False
            return True

        def friends(pid, p):
            return sum(1 for k, f in self.foes.items() if k != pid and f.get("pos") and t - f["t"] < 10
                       and math.dist(f["pos"][:2], p[:2]) < GROUP_R)

        # 1) our target is visible: close in unless it's sitting in a group
        if target is not None:
            p = target["pos"]
            if friends(target["id"], p) >= 2:
                return None
            if target["_d"] <= CLOSE_IN + 4:
                return "here"
            return (p[0], p[1]) if ok(p) else None
        # 2) someone we saw recently: weakest / most isolated / closest
        best, bs = None, 1e18
        for k, f in self.foes.items():
            if not f.get("pos") or t - f["t"] > 15 or not ok(f["pos"]):
                continue
            p = f["pos"]
            s = math.dist(p[:2], (x, y)) + 90 * friends(k, p) + 0.05 * f.get("hp", 1400)
            if s < bs:
                best, bs = (p[0], p[1]), s
        if best is not None and bs < 400:
            return best
        # 3) follow the sound of fighting or footsteps
        snd = [s for s in obs.get("sounds", []) if s["kind"] in
               ("gun", "laser", "lightning", "mortar", "step", "boost", "jet", "reload", "laser_charge")]
        if snd:
            s = min(snd, key=lambda s: {"near": 0, "mid": 1, "far": 2}[s["dist"]])
            d = {"near": 40.0, "mid": 110.0, "far": 200.0}[s["dist"]]
            p = (x + d * math.cos(s["bearing"]), y + d * math.sin(s["bearing"]))
            if ok(p):
                self.sound_goal, self.sound_goal_t = p, t
        sg = getattr(self, "sound_goal", None)
        if sg is not None and t - self.sound_goal_t < 6 and math.hypot(sg[0] - x, sg[1] - y) > 8:
            return sg
        return None

    # ------------------------------------------------------------------ pieces
    def loot_goal(self, obs, me, x, y, zf, z, t, enemies):
        """Pick something worth walking to: drop, better weapon, cube, crate, NPC. None = nothing."""
        nav = self.nav
        ncx, ncy = z["next_center"]
        rn = z["next_radius"]
        time_left = max(0.0, z["shrink_end"] - t)

        def ok(p, maxd):
            d = math.hypot(p[0] - x, p[1] - y)
            if d > maxd:
                return False
            if math.hypot(p[0] - ncx, p[1] - ncy) > rn * 0.95 and time_left < 30:
                return False
            if p[2] - nav.ground_at(p[0], p[1]) > 3.0:  # on a roof
                return False
            return True

        # supply drop: full HP and 3 cubes
        for dr in obs.get("drops", []):
            p = (dr["pos"][0], dr["pos"][1], nav.ground_at(dr["pos"][0], dr["pos"][1]))
            d = math.hypot(p[0] - x, p[1] - y)
            eta = d / me["stats"]["walk_speed"]
            other = dr.get("holder") not in (None, me["id"])
            if d < 160 and eta + 3 > dr.get("lands_in", 0) and not other:
                return (p[0], p[1])
        # better weapon
        slots = me["weapons"]
        have = {w["kind"] for w in slots if w}
        worst = 99 if slots[1] is None else WEAPON_RANK.get(slots[1]["kind"], 0)
        best = None
        for pk in obs.get("pickups", []):
            if pk["kind"] in have or self.loot_ignore.get(pk["id"], -1) > t:
                continue
            if slots[1] is not None and WEAPON_RANK.get(pk["kind"], 0) <= worst:
                continue
            reach = 130 if (slots[1] is None and t < 90) else 70   # early on, go out of our way for a weapon
            if not ok(pk["pos"], reach):
                continue
            d = math.dist(pk["pos"][:2], (x, y))
            if best is None or d < best[0]:
                best = (d, pk)
        if best:
            return (best[1]["pos"][0], best[1]["pos"][1])
        # power cubes
        cs = [c for c in obs.get("cubes", []) if ok(c["pos"], 70)]
        if cs:
            c = min(cs, key=lambda c: math.dist(c["pos"][:2], (x, y)))
            return (c["pos"][0], c["pos"][1])
        return None

    def weapons(self, obs, me, x, y, zf, target, npc_target, mode, t):
        """Returns (aim or None, fire[2], extra action fields)."""
        st = me["stats"]
        wcfg = self.cfg["weapons"]
        slots = me["weapons"]
        fire = [False, False]
        extra = {}
        nav = self.nav

        # pick up a weapon we're standing on
        have = {w["kind"] for w in slots if w}
        for pk in obs.get("pickups", []):
            if pk["kind"] in have or self.loot_ignore.get(pk["id"], -1) > t:
                continue
            if math.dist(pk["pos"][:2], (x, y)) < self.cfg["loot"]["pickup_radius"] - 0.3 and abs(pk["pos"][2] - zf) < 3.5:
                if slots[1] is None:
                    extra["take"] = pk["id"]
                elif WEAPON_RANK.get(pk["kind"], 0) > WEAPON_RANK.get(slots[1]["kind"], 0) and \
                        not (slots[1]["kind"] == "laser" and slots[1].get("charge", 0) > 0):
                    extra["take"] = pk["id"]
                    extra["drop"] = 1
                break

        # what are we shooting at?
        tgt = target or npc_target
        crate = None
        if tgt is None and mode in ("hold", "rotate", "loot"):
            cs = [c for c in obs.get("crates", []) if math.dist(c["pos"], me["pos"]) < 90]
            if cs:
                crate = min(cs, key=lambda c: math.dist(c["pos"], me["pos"]))
        if tgt is None and crate is None:
            # nothing to shoot: top up the magazine
            g = slots[0]
            if g and g["kind"] == "gun" and g.get("reload", 0) == 0 and g.get("ammo", 99) < g["stats"]["mag"] * 0.5:
                extra["reload"] = True
            # let go of a charging laser rather than waste it? keep it; it fires on its own
            return None, fire, extra

        if tgt is not None:
            tpos, tvel = tgt["pos"], tgt.get("vel", [0, 0, 0])
            height = 4.5 * (1 + 0.05 * tgt.get("cubes", 0)) if tgt["kind"] == "mech" else 1.8
            radius = 2.5 if tgt["kind"] == "mech" else 1.8
        else:
            tpos, tvel, height, radius = crate["pos"], [0, 0, 0], 1.2, 1.4
        dist = math.dist(tpos[:2], (x, y))
        muzzle = st.get("muzzle_height", 7)
        clear = nav.los((x, y, zf + muzzle), (tpos[0], tpos[1], tpos[2] + height))

        gun_i = next((i for i, w in enumerate(slots) if w and w["kind"] == "gun"), None)
        alt_i = next((i for i, w in enumerate(slots) if w and w["kind"] != "gun"), None)
        alt = slots[alt_i] if alt_i is not None else None

        aim = None
        # gun solution
        g_aim = None
        if gun_i is not None:
            gw = slots[gun_i]
            g_aim, _ = self.lead(me["pos"], muzzle, tpos, tvel, gw["stats"]["speed"], height)
        # special weapon
        if alt is not None:
            ws = alt["stats"]
            k = alt["kind"]
            if k == "lightning":
                if dist < ws.get("range", 45) * 0.92:
                    aim, _ = self.lead(me["pos"], muzzle, tpos, tvel, 1e9, height)
                    cone = math.radians(ws.get("cone_deg", 30)) * 0.4
                    if abs(ang_diff(me["aim"][0], aim[0])) < cone and me["energy"] > 8 and alt["cooldown"] <= 0.05:
                        fire[alt_i] = True
            elif k == "laser":
                charging = alt.get("charge", 0) > 0
                if dist < ws.get("range", 280) * 0.95 and clear and mode != "retreat" and (alt["cooldown"] <= 0.3 or charging):
                    aim, _ = self.lead(me["pos"], muzzle, tpos, tvel, 1e9, height)
                    full = ws.get("charge", 0.75)
                    if charging and alt["charge"] >= full * 0.97 and self.aim_ok(me, aim, dist, radius):
                        fire[alt_i] = False  # let go: fire
                    else:
                        fire[alt_i] = alt["cooldown"] <= 0.05 or charging
                elif charging:
                    fire[alt_i] = False
            elif k == "mortar":
                d3 = math.dist(tpos, me["pos"])
                gun_busy = gun_i is None or slots[gun_i].get("reload", 0) > 0 or dist > 200 or not clear
                if alt["cooldown"] <= 0.05 and 25 < d3 < 220 and gun_busy:
                    m_aim = self.mortar_aim(me["pos"], st.get("height", 9) + 0.5, tpos, tvel,
                                            ws.get("speed", 72), ws.get("gravity", wcfg["mortar"]["gravity"]))
                    if m_aim:
                        aim = m_aim
                        if self.aim_ok(me, aim, dist, 6.0):
                            fire[alt_i] = True
        if aim is None:
            aim = g_aim
        if aim is None:
            return None, fire, extra
        if gun_i is not None and g_aim is not None and clear:
            gw = slots[gun_i]
            if dist < gw["stats"]["range"] and gw.get("reload", 0) == 0 and self.aim_ok(me, g_aim, dist, radius, 1.2):
                fire[gun_i] = True
        # resting from a retreat: don't shoot unless they're close (shooting stops regen)
        if mode == "retreat" and tgt is not None and tgt.get("hp", 1400) >= 160 and t - self.last_hit_t > 1.0:
            fire = [False, False] if not (alt and alt["kind"] == "laser" and alt.get("charge", 0) > 0) else fire
        return aim, fire, extra

    def dodge(self, obs, me, x, y, zf):
        # a charging laser pointed at us
        for m in obs.get("mechs", []):
            if m.get("kind") == "mech" and m.get("charging"):
                to_me = math.atan2(y - m["pos"][1], x - m["pos"][0])
                d = math.dist(m["pos"][:2], (x, y))
                if abs(ang_diff(m["aim"][0], to_me)) < max(0.05, math.atan2(4, max(d, 1))) and d < 300:
                    px_, py_ = -math.sin(to_me), math.cos(to_me)
                    s = 1 if self.nav.free_xy(x + px_ * 4, y + py_ * 4) else -1
                    return (px_ * s, py_ * s)
        # mortar shells that will land near us
        g = self.cfg["weapons"]["mortar"]["gravity"]
        for p in obs.get("projectiles", []):
            if p.get("kind") != "mortar":
                continue
            px_, py_, pz = p["pos"]
            vx, vy, vz = p["vel"]
            # time to fall to our height
            a, b, c = -0.5 * g, vz, pz - zf
            disc = b * b - 4 * a * c
            if disc < 0:
                continue
            tl = (-b - math.sqrt(disc)) / (2 * a)
            if tl <= 0 or tl > 5:
                continue
            lx, ly = px_ + vx * tl, py_ + vy * tl
            if math.hypot(lx - x, ly - y) < 11:
                return unit(x - lx, y - ly) if math.hypot(lx - x, ly - y) > 0.5 else unit(-vy, vx)
        return None

    def look(self, obs, me, mv, t):
        """Where to point the torso (and eyes) when not shooting."""
        since_hit = t - self.last_hit_t
        if since_hit < 3 and self.hit_bearing is not None:
            return [self.hit_bearing, 0.0]
        loud = [s for s in obs.get("sounds", []) if s["kind"] in ("gun", "laser", "laser_charge", "lightning", "mortar", "step", "boost", "jet", "reload")
                and s.get("dist") in ("near", "mid")]
        if loud:
            s = min(loud, key=lambda s: (s["dist"] != "near", s.get("age", 0)))
            return [s["bearing"], 0.0]
        base = math.atan2(mv[1], mv[0]) if mv != (0.0, 0.0) else me["aim"][0]
        if mv == (0.0, 0.0):
            return [me["aim"][0] + 0.9, 0.0]  # standing still: keep turning to scan all around
        return [base + 0.9 * math.sin(t * 1.3), 0.0]


if __name__ == "__main__":
    mf.run(Bot)
