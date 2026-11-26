"""CARLA world: finds a real junction in the map with two straight, crossing
movements, builds both paths from map waypoints, and drives two vehicles along
them in synchronous mode (50 ms ticks).

Vehicles are moved kinematically with set_transform (physics off), like the
scripted controllers in LAST-V2X: speeds come from the IMA logic and the
behavior profile, positions and geometry come from the CARLA map.
"""
from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple

try:
    import carla
except ImportError as e:  # pragma: no cover
    raise ImportError("The 'carla' Python package is not installed. See README step 3.") from e

from .scenario import Scenario

Point = Tuple[float, float, float, float]  # x, y, z, yaw(deg)


def _yaw_diff(a: float, b: float) -> float:
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def _seg_intersect(p1, p2, p3, p4):
    """Intersection of segments p1p2 and p3p4 in 2D; returns (u, w) fractions or None."""
    x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-9:
        return None
    u = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    w = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
    if 0 <= u <= 1 and 0 <= w <= 1:
        return u, w
    return None


class Path:
    def __init__(self, pts: List[Point]):
        self.pts = pts
        self.cum = [0.0]
        for a, b in zip(pts, pts[1:]):
            self.cum.append(self.cum[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
        self.length = self.cum[-1]

    def at(self, s: float) -> Point:
        s = min(max(s, 0.0), self.length)
        lo, hi = 0, len(self.cum) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.cum[mid] <= s:
                lo = mid
            else:
                hi = mid
        a, b = self.pts[lo], self.pts[hi]
        seg = self.cum[hi] - self.cum[lo]
        f = 0.0 if seg < 1e-9 else (s - self.cum[lo]) / seg
        yaw = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) if seg > 1e-9 else a[3]
        return (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]), yaw)


def _intersect_paths(pa: Path, pb: Path, center=None, radius=40.0) -> Optional[Tuple[float, float]]:
    def near(p):
        return center is None or math.hypot(p[0] - center[0], p[1] - center[1]) < radius
    ia = [i for i in range(len(pa.pts) - 1) if near(pa.pts[i])]
    ib = [j for j in range(len(pb.pts) - 1) if near(pb.pts[j])]
    for i in ia:
        for j in ib:
            r = _seg_intersect(pa.pts[i][:2], pa.pts[i + 1][:2], pb.pts[j][:2], pb.pts[j + 1][:2])
            if r:
                u, w = r
                sa = pa.cum[i] + u * (pa.cum[i + 1] - pa.cum[i])
                sb = pb.cum[j] + w * (pb.cum[j + 1] - pb.cum[j])
                return sa, sb
    return None


class CarlaWorld:
    name = "carla"

    def __init__(self, host="localhost", port=2000, town="Town05", watch=False, dt=0.05,
                 approach_len=220.0, exit_len=40.0, junction_id: Optional[int] = None):
        self.client = carla.Client(host, port)
        self.client.set_timeout(30.0)
        world = self.client.get_world()
        if not world.get_map().name.endswith(town):
            print(f"[carla] loading {town} ...")
            world = self.client.load_world(town)
        self.world = world
        self.map = world.get_map()
        self.watch = watch
        self.dt = dt
        self._orig_settings = world.get_settings()
        st = world.get_settings()
        st.synchronous_mode = True
        st.fixed_delta_seconds = dt
        st.no_rendering_mode = not watch
        world.apply_settings(st)
        self.approach_len = approach_len
        self.exit_len = exit_len
        self.ego_path, self.cross_path, self.ego_cp, self.cross_cp, self.jid = self._find_crossing(junction_id)
        print(f"[carla] using junction {self.jid}: ego conflict at s={self.ego_cp:.1f} m, "
              f"crossing conflict at s={self.cross_cp:.1f} m")
        bpl = world.get_blueprint_library()
        self.bp_ego = self._bp(bpl, "vehicle.lincoln.mkz_2020")
        self.bp_cross = self._bp(bpl, "vehicle.tesla.model3")
        self.bp_ego.set_attribute("role_name", "ego")
        if self.bp_ego.has_attribute("color"):
            self.bp_ego.set_attribute("color", "0,90,200")
        if self.bp_cross.has_attribute("color"):
            self.bp_cross.set_attribute("color", "200,30,30")
        self.actors: List = []

    @staticmethod
    def _bp(bpl, name):
        f = bpl.filter(name)
        return f[0] if f else bpl.filter("vehicle.*")[0]

    # ---------- path construction from map waypoints ----------
    def _back(self, wp, length, step=2.0) -> List[Point]:
        pts, d = [], 0.0
        cur = wp
        while d < length:
            prev = cur.previous(step)
            if not prev:
                break
            cur = min(prev, key=lambda w: _yaw_diff(w.transform.rotation.yaw, cur.transform.rotation.yaw))
            l = cur.transform.location
            pts.append((l.x, l.y, l.z, cur.transform.rotation.yaw))
            d += step
        return list(reversed(pts))

    def _forward(self, wp, length, step=1.0, until=None) -> List[Point]:
        pts, d, cur = [], 0.0, wp
        while d < length:
            nxt = cur.next(step)
            if not nxt:
                break
            cur = min(nxt, key=lambda w: _yaw_diff(w.transform.rotation.yaw, cur.transform.rotation.yaw))
            l = cur.transform.location
            pts.append((l.x, l.y, l.z, cur.transform.rotation.yaw))
            d += step
            if until is not None and l.distance(until) < step:
                break
        return pts

    def _lane_path(self, start_wp, end_wp) -> Path:
        l0 = start_wp.transform.location
        pts = self._back(start_wp, self.approach_len)
        pts.append((l0.x, l0.y, l0.z, start_wp.transform.rotation.yaw))
        pts += self._forward(start_wp, 80.0, until=end_wp.transform.location)
        pts += self._forward(end_wp, self.exit_len, step=2.0)
        return Path(pts)

    def _find_crossing(self, junction_id):
        seen, junctions = set(), []
        for wp in self.map.generate_waypoints(3.0):
            if wp.is_junction:
                j = wp.get_junction()
                if j.id not in seen:
                    seen.add(j.id)
                    junctions.append(j)
        if junction_id is not None:
            junctions = [j for j in junctions if j.id == junction_id]
        best = None
        for j in junctions:
            lanes = j.get_waypoints(carla.LaneType.Driving)
            straight = [(a, b) for (a, b) in lanes
                        if _yaw_diff(a.transform.rotation.yaw, b.transform.rotation.yaw) < 15]
            for i in range(len(straight)):
                for k in range(i + 1, len(straight)):
                    a, b = straight[i], straight[k]
                    ang = _yaw_diff(a[0].transform.rotation.yaw, b[0].transform.rotation.yaw)
                    if not (70 <= ang <= 110):
                        continue
                    pa, pb = self._lane_path(*a), self._lane_path(*b)
                    c = j.bounding_box.location
                    hit = _intersect_paths(pa, pb, center=(c.x, c.y))
                    if not hit:
                        continue
                    sa, sb = hit
                    room = min(sa, sb)  # how far back each approach goes
                    if best is None or room > best[0]:
                        best = (room, pa, pb, sa, sb, j.id)
            if best and best[0] >= 200:   # enough approach road for distances up to 200 m
                break
        if best is None:
            raise RuntimeError("No junction with two straight crossing movements found in this map.")
        if best[0] < 150:
            print(f"[carla] WARNING: longest approach only {best[0]:.0f} m; large start distances will be clamped.")
        _, pa, pb, sa, sb, jid = best
        return pa, pb, sa, sb, jid

    # ---------- per-run interface ----------
    def reset(self, scn: Scenario, ghost: bool = False):
        self._destroy()
        self.ego_s0 = self.ego_cp - scn.ego_dist
        self.cross_s0 = self.cross_cp - scn.cross_dist
        if self.ego_s0 < 0 or self.cross_s0 < 0:
            print(f"[carla] WARNING: start distance longer than available road; clamping ({scn.name})")
            self.ego_s0, self.cross_s0 = max(self.ego_s0, 0), max(self.cross_s0, 0)
        self.ego_cp_s = self.ego_cp - self.ego_s0
        self.cross_cp_s = self.cross_cp - self.cross_s0
        self.ego = self._spawn(self.bp_ego, self.ego_path.at(self.ego_s0))
        self.cross = self._spawn(self.bp_cross, self.cross_path.at(self.cross_s0))
        if self.watch:
            loc = carla.Location(*self.ego_path.at(self.ego_cp)[:3])
            self.world.get_spectator().set_transform(
                carla.Transform(loc + carla.Location(z=45), carla.Rotation(pitch=-90)))
        self.world.tick()

    def _spawn(self, bp, p: Point, z_off=0.0):
        tf = carla.Transform(carla.Location(x=p[0], y=p[1], z=p[2] + 0.3 + z_off), carla.Rotation(yaw=p[3]))
        actor = self.world.try_spawn_actor(bp, tf)
        if actor is None:   # spawn point blocked (e.g., parked car): lift it up and retry
            tf.location.z += 2.0
            actor = self.world.spawn_actor(bp, tf)
        actor.set_simulate_physics(False)
        self.actors.append(actor)
        return actor

    def step(self, ego_s: float, cross_s: float):
        pe = self.ego_path.at(self.ego_s0 + ego_s)
        pc = self.cross_path.at(self.cross_s0 + cross_s)
        self.ego.set_transform(carla.Transform(carla.Location(x=pe[0], y=pe[1], z=pe[2] + 0.05), carla.Rotation(yaw=pe[3])))
        self.cross.set_transform(carla.Transform(carla.Location(x=pc[0], y=pc[1], z=pc[2] + 0.05), carla.Rotation(yaw=pc[3])))
        self.world.tick()
        if self.watch:
            time.sleep(self.dt)  # roughly real time, so you can screen-record
        # read the true positions back from CARLA
        le, lc = self.ego.get_location(), self.cross.get_location()
        return (le.x, le.y), (lc.x, lc.y)

    def _destroy(self):
        for a in self.actors:
            try:
                a.destroy()
            except Exception:
                pass
        self.actors = []

    def close(self):
        self._destroy()
        try:
            self.world.apply_settings(self._orig_settings)
        except Exception:
            pass
