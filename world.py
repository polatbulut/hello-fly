"""A persistent odour world for the closed-loop fly.

WHAT FLYGYM ALREADY GIVES US (verified against the installed 1.2.1 source,
site-packages/flygym/arena/sensory_environment.py):

  * OdorArena ALREADY supports N sources and K odour dimensions. `odor_source`
    is (n_sources, 3) and `peak_odor_intensity` is (n_sources, n_dims); the
    only constraint enforced is that their first axes match (line 111-114).
    flygym/tests/test_odor.py::test_odor_dimensions builds 5 sources x 4 dims
    and asserts the observation is (n_dims, 4). So a "MultiOdorArena" subclass
    would be pure ceremony. We do NOT write one.

  * get_olfaction() is called live, every physics step, from
    Fly.get_observation (fly.py:1261-1264). NOTHING about the odour field is
    baked into the MuJoCo model -- the only MJCF artefacts are the capsule
    marker geoms, which are cosmetic. Therefore the field can be mutated
    between steps with no model rebuild. Depletion is possible.

WHAT WE ADD (and why a subclass IS justified here):
  1. An anisotropic, wind-shaped plume. OdorArena's `diffuse_func` receives
     only the scalar Euclidean distance (line 182: `self.diffuse_func(dist_euc)`),
     so it CANNOT express direction. A non-radial plume requires overriding
     get_olfaction. That is the real reason to subclass.
  2. Depletion / regrowth, with the cache-coherency trap handled (see
     set_reserve).
  3. Periodic (toroidal) odour so a long-lived fly never runs out of world.
  4. A cheaper get_olfaction: the stock one repeats the source positions along
     the odour-dimension axis and computes the SAME distance k times
     (lines 135-141, 180-181). We compute distances once, shape (n, w), and
     broadcast. Bit-identical arithmetic, k times less of it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from flygym.arena import OdorArena

__all__ = ["Source", "LivingWorldArena", "make_world", "world_camera_params"]


# --------------------------------------------------------------------------- #
#  Source description
# --------------------------------------------------------------------------- #
@dataclass
class Source:
    """One odour source in the world.

    pos       : (x, y, z) in mm. z is where the marker sits; it also enters the
                vertical Gaussian term, so keep it near the ground (1-2 mm).
    intensity : length-K vector of peak intensity, one entry per odour
                DIMENSION. Dimension 0 is the "attractive" channel by
                convention in this repo; dimension 1 is the "aversive" channel.
                Nothing in FlyGym or in the connectome knows that -- see
                bridge_valence.py. The convention is ours.
    rgba      : marker colour.
    edible    : if False the source never depletes (you do not eat a repellent).
    label     : for plots.
    """
    pos: tuple[float, float, float]
    intensity: Sequence[float]
    rgba: tuple[float, float, float, float] = (0.2, 0.7, 0.3, 1.0)
    edible: bool = True
    label: str = "src"


# --------------------------------------------------------------------------- #
#  The arena
# --------------------------------------------------------------------------- #
class LivingWorldArena(OdorArena):
    """OdorArena with a wind-shaped plume, depletable sources, and a torus.

    Parameters
    ----------
    sources : list[Source]
    period : float or None
        Side of the square odour unit cell, in mm. If not None the odour field
        is evaluated with the minimum-image convention, i.e. the world is a
        torus of this size. The fly is NEVER teleported -- it walks off into
        absolute coordinates forever on an (optionally infinite) ground plane,
        and the odour it smells is periodic. This is the whole trick: no
        wrap, no respawn, no discontinuity in the brain's input, no
        re-solving of contacts, and an unbounded lifetime.
    ground_half : float
        Half-size of the rendered checkerboard, in mm. 0 means MuJoCo renders
        an INFINITE plane. Collision is unaffected either way: MuJoCo planes
        are laterally infinite for collision regardless of `size` (verified in
        engine_collision_driver.c's plane broadphase filter, which only
        projects onto the plane normal, and in engine_collision_primitive.c's
        mjc_PlaneSphere/Capsule/Box, none of which test size[0]/size[1]).
    wind : (wx, wy)
        Wind direction in the xy plane. Normalised internally. The plume
        extends DOWNWIND of each source, i.e. toward +wind.
    wind_speed : float
        Enters the Gaussian-plume denominator (mm/s). Only sets overall scale,
        which `calibrate_gain` then absorbs; kept so the formula is the real
        one and not a lookalike.
    sigma0, spread_y, spread_z : float
        Plume width at the source and its linear growth downwind:
        sigma_y(s) = sigma0 + spread_y * s.
    near_gain, upwind_decay : float
        Near-field isotropic term. Without it the fly is olfactorily BLIND
        anywhere upwind of a source, because a pure Gaussian plume is exactly
        zero there -- it would walk past the source and never turn back. The
        term is near_gain / r^2 damped by exp(-upwind_distance/upwind_decay),
        so close to the source the field still looks like the r^-2 the
        existing bridge constants were calibrated against.

        upwind_decay is the single most behaviourally consequential knob in
        this class. MEASURED at 40 mm upwind of a source:
            upwind_decay =  6 mm -> exp(-40/6)  = 0.0013 of the r^-2 value
            upwind_decay = 18 mm -> exp(-40/18) = 0.11
        The bridge does pure gradient climbing -- it has no casting and no
        upwind surge, which is how real flies solve exactly this problem. At
        6 mm a fly that starts upwind of every source simply never finds one
        and the run is a null result about the arena, not about the brain.
        Default 18 mm keeps the field strongly anisotropic (9x) while leaving
        the problem solvable. Set 6 mm deliberately if hard mode is the point.
    deplete_tau_s : float
        Time constant of reserve consumption while the fly is within
        feed_radius. reserve -= dt / deplete_tau_s.
    regrow_tau_s : float or None
        Exponential recovery toward full: reserve += (1-reserve)*dt/tau.
        None disables regrowth (sources are finite and the world runs down).
    feed_radius : float
        mm. Within this distance of a source centre the fly is "feeding".
    reserve_floor : float
        Lower clamp on reserve, in [0, 1]. A floor > 0 keeps the marker
        faintly smellable so a depleted patch is still navigable rather than a
        hole in the world.
    tile_radius : int
        Markers are replicated on a (2*tile_radius+1)^2 lattice so the
        periodic images are visible. They are re-anchored around the fly's
        current cell by `retile_markers`, so cost is constant however far the
        fly walks.
    collide_markers : bool
        Default False. The stock OdorArena adds marker capsules with MuJoCo's
        default contype/conaffinity=1, so with the repo's current
        `odor_source z=1.5, marker_size=0.3` the capsule's underside sits at
        z = 1.5 - 0.3 - 0.3 = 0.9 mm -- squarely in the path of the fly's head
        and antennae. Over a 3 s episode you may never notice. Over an hour of
        repeated visits to a source you will. We turn collision off.
    """

    def __init__(
        self,
        sources: Sequence[Source],
        period: Optional[float] = 300.0,
        ground_half: float = 0.0,
        friction: tuple[float, float, float] = (1, 0.005, 0.0001),
        num_sensors: int = 4,
        wind: tuple[float, float] = (1.0, 0.0),
        wind_speed: float = 100.0,
        sigma0: float = 2.5,
        spread_y: float = 0.22,
        spread_z: float = 0.09,
        near_gain: float = 1.0,
        upwind_decay: float = 18.0,
        deplete_tau_s: float = 20.0,
        regrow_tau_s: Optional[float] = 120.0,
        feed_radius: float = 3.0,
        reserve_floor: float = 0.05,
        tile_radius: int = 2,
        marker_size: float = 0.3,
        collide_markers: bool = False,
        calibrate_at_mm: Optional[float] = 30.0,
    ):
        self.sources = list(sources)
        n = len(self.sources)
        if n == 0:
            raise ValueError("a world needs at least one source")

        pos = np.array([s.pos for s in self.sources], dtype=float)          # (n,3)
        peak0 = np.array([s.intensity for s in self.sources], dtype=float)  # (n,k)
        if peak0.ndim != 2:
            raise ValueError("every Source.intensity must be the same length K")
        colors = [tuple(s.rgba) for s in self.sources]

        # Must exist BEFORE super().__init__: OdorArena.__init__ reads
        # self.odor_dimensions at line 137 to build its repeat caches, and we
        # override that property to read n_dims.
        self.n_sources = n
        self.n_dims = peak0.shape[1]

        # The parent builds the ground, the lights and the markers, and stores
        # odor_source / peak_odor_intensity / _peak_intensity_repeated. We pass
        # marker_colors explicitly because OdorArena's default-colour branch
        # indexes color_cycle_rgb[i % num_odor_sources] (line 123) instead of
        # [i % len(color_cycle_rgb)], which raises IndexError for >10 sources.
        super().__init__(
            size=(ground_half, ground_half),
            friction=friction,
            num_sensors=num_sensors,
            odor_source=pos,
            peak_odor_intensity=peak0,
            diffuse_func=lambda x: x ** -2,   # never called; we override get_olfaction
            marker_colors=colors,
            marker_size=marker_size,
        )

        self._pos = pos
        self._peak0 = peak0
        self._peak = peak0.copy()

        self.period = None if period is None else float(period)
        w = np.asarray(wind, dtype=float)
        nw = np.linalg.norm(w)
        if nw < 1e-12:
            raise ValueError("wind must be a non-zero 2-vector")
        self._wind = w / nw
        self.wind_speed = float(wind_speed)
        self.sigma0 = float(sigma0)
        self.spread_y = float(spread_y)
        self.spread_z = float(spread_z)
        self.near_gain = float(near_gain)
        self.upwind_decay = float(upwind_decay)
        self.gain = 1.0

        self.deplete_tau_s = float(deplete_tau_s)
        self.regrow_tau_s = None if regrow_tau_s is None else float(regrow_tau_s)
        self.feed_radius = float(feed_radius)
        self.reserve_floor = float(reserve_floor)
        self.edible = np.array([s.edible for s in self.sources], dtype=bool)

        # --- live world state -------------------------------------------------
        self.reserve = np.ones(n)          # in [reserve_floor, 1]
        self.consumed = np.zeros(n)        # cumulative reserve-seconds eaten
        self.visits = np.zeros(n, dtype=np.int64)
        self.dwell_s = np.zeros(n)
        self._was_feeding = np.zeros(n, dtype=bool)
        self.curr_time = 0.0
        self._countdown = 0        # see get_olfaction
        self._cache = None

        # --- intermittency (OFF by default; see note in the docstring of
        #     set_intermittency) ---------------------------------------------
        self.intermittency = 0.0
        self.filament_mm = 25.0
        self._phase = np.linspace(0.0, 2 * np.pi, n, endpoint=False)

        # --- markers ----------------------------------------------------------
        self._tile_radius = int(tile_radius)
        self._marker_bodies: list[list] = [[] for _ in range(n)]
        self._marker_geoms: list[list] = [[] for _ in range(n)]
        self._tile_offsets: list[tuple[int, int]] = [(0, 0)]
        self._base_rgba = np.array(colors, dtype=float)
        self._rgba_last = np.full(n, -1.0)
        self._tile_origin = np.zeros(2)
        self._collect_stock_markers()
        if self.period is not None and self._tile_radius > 0:
            self._add_tile_markers(marker_size)
        if not collide_markers:
            for geoms in self._marker_geoms:
                for g in geoms:
                    g.contype = 0
                    g.conaffinity = 0

        if calibrate_at_mm is not None:
            self.calibrate_gain(float(calibrate_at_mm))

    # ------------------------------------------------------------------ markers
    def _collect_stock_markers(self):
        for i in range(self.n_sources):
            body = self.root_element.find("body", f"odor_source_marker_{i}")
            self._marker_bodies[i].append(body)
            self._marker_geoms[i].extend(body.find_all("geom"))

    def _add_tile_markers(self, marker_size):
        R = self._tile_radius
        offsets = [(ix, iy)
                   for ix in range(-R, R + 1)
                   for iy in range(-R, R + 1)
                   if not (ix == 0 and iy == 0)]
        # _tile_offsets must stay parallel to _marker_bodies[i], whose element
        # 0 is the stock marker created by OdorArena at offset (0, 0).
        self._tile_offsets = [(0, 0)] + offsets
        for i, src in enumerate(self.sources):
            for ix, iy in offsets:
                p = (src.pos[0] + ix * self.period,
                     src.pos[1] + iy * self.period,
                     src.pos[2])
                b = self.root_element.worldbody.add(
                    "body", name=f"odor_tile_{i}_{ix + R}_{iy + R}",
                    pos=p, mocap=True)
                g = b.add("geom", type="capsule",
                          size=(marker_size, marker_size), rgba=src.rgba)
                self._marker_bodies[i].append(b)
                self._marker_geoms[i].append(g)

    def retile_markers(self, fly_xy, physics):
        """Re-anchor the marker lattice around the fly's current unit cell.

        Call at render cadence, not per step. `physics.bind(body).mocap_pos` is
        the documented dm_control path for a body declared with mocap=True
        (dm_control/mjcf/physics.py:663 maps such a body to the 'mocap_body'
        namespace; dm_control/mjcf/physics_test.py:185 assigns mocap_pos).
        """
        if self.period is None or self._tile_radius == 0:
            return
        L = self.period
        origin = np.floor(np.asarray(fly_xy, dtype=float) / L + 0.5) * L
        if np.allclose(origin, self._tile_origin):
            return
        self._tile_origin = origin
        for i, src in enumerate(self.sources):
            for body, (ix, iy) in zip(self._marker_bodies[i], self._tile_offsets):
                physics.bind(body).mocap_pos = (
                    src.pos[0] + origin[0] + ix * L,
                    src.pos[1] + origin[1] + iy * L,
                    src.pos[2],
                )

    def refresh_markers(self, physics, tol=0.02):
        """Fade each marker's alpha with its remaining reserve.

        Guarded by `tol` because this touches every tiled marker geom (75 of
        them at tile_radius=2 with 3 sources) through dm_control's `bind`,
        which is not free. With deplete_tau_s=20 a 2% change takes 0.4 s of
        feeding, so in practice this fires a few times per visit rather than
        at every render.
        """
        for i in range(self.n_sources):
            if abs(self.reserve[i] - self._rgba_last[i]) < tol:
                continue
            self._rgba_last[i] = self.reserve[i]
            rgba = self._base_rgba[i].copy()
            rgba[3] = 0.12 + 0.88 * float(self.reserve[i])
            for g in self._marker_geoms[i]:
                physics.bind(g).rgba = rgba

    # ------------------------------------------------------- field mathematics
    def _displacement(self, q):
        """(w,3) sensor positions -> (n,w,3) minimum-image displacements."""
        d = q[None, :, :] - self._pos[:, None, :]
        if self.period is not None:
            d[:, :, :2] -= self.period * np.round(d[:, :, :2] / self.period)
        return d

    def _shape(self, d):
        """(n,w,3) displacement -> (n,w) dimensionless field shape.

        Gaussian plume (Pasquill-Gifford form) downwind, plus an isotropic
        near-field term that survives upwind.
        """
        dxy = d[:, :, :2]
        s = dxy @ self._wind                                   # (n,w) downwind
        cross = dxy - s[:, :, None] * self._wind[None, None, :]
        c = np.sqrt(cross[:, :, 0] ** 2 + cross[:, :, 1] ** 2)  # (n,w) crosswind
        h = np.abs(d[:, :, 2])
        r2 = d[:, :, 0] ** 2 + d[:, :, 1] ** 2 + d[:, :, 2] ** 2

        s_pos = np.maximum(s, 0.0)
        sig_y = self.sigma0 + self.spread_y * s_pos
        sig_z = self.sigma0 + self.spread_z * s_pos
        plume = np.exp(-0.5 * (c / sig_y) ** 2 - 0.5 * (h / sig_z) ** 2)
        plume /= (2.0 * np.pi * sig_y * sig_z * self.wind_speed)
        plume = np.where(s > 0.0, plume, 0.0)

        near = self.near_gain / np.maximum(r2, 1e-6)
        near *= np.exp(-np.maximum(-s, 0.0) / self.upwind_decay)

        shape = self.gain * (plume + near)

        if self.intermittency > 0.0:
            # Advected filaments: a travelling wave in the downwind coordinate,
            # confined to the plume's own width. Deterministic and allocation
            # free. OFF by default -- see set_intermittency.
            k = 2.0 * np.pi / self.filament_mm
            env = np.exp(-0.5 * (c / sig_y) ** 2)
            osc = 0.5 * (1.0 + np.cos(
                k * (s - self.wind_speed * self.curr_time)
                + self._phase[:, None]))
            m = 1.0 - self.intermittency * env * (1.0 - osc)
            shape = shape * m
        return shape

    # --------------------------------------------------------------- readout
    def begin_exchange(self, n_physics_steps):
        """Tell the arena how many physics steps until the loop next READS the
        observation. See the note on _countdown in get_olfaction."""
        self._countdown = int(n_physics_steps)

    def get_olfaction(self, antennae_pos: np.ndarray) -> np.ndarray:
        """(w,3) sensor positions -> (K, w) intensities. Same contract as
        OdorArena.get_olfaction, which returns intensity.sum(axis=0) of shape
        (k, w) (sensory_environment.py:184).

        MEASURED COST, and why the countdown exists:

          this field, evaluated in full   50.1 us / call
          stock OdorArena r^-2            10.7 us / call

        and Fly.get_observation calls this on EVERY physics step
        (fly.py:1261-1264), i.e. 10,000 times per simulated second, which is
        0.50 s of wall per simulated second. The closed loop READS the
        observation once per 5 ms exchange -- 200 times per second. 49 of
        every 50 evaluations are computed and thrown away. (The same waste
        exists with the stock arena; it is just smaller, 0.11 s.)

        Optimising the arithmetic does not help: a fused, buffer-reusing,
        einsum-contracting rewrite measured 45.1 us against 50.1 us. On (3,4)
        arrays the time is numpy per-operation dispatch overhead, not flops.

        So instead we skip the calls nobody reads. `begin_exchange(n)` arms a
        countdown; the full field is evaluated on the LAST step before the read
        and the cached value is returned on the others. The value the loop
        actually consumes is bit-for-bit what it would have been. If the
        countdown is not armed (someone else is driving the sim) every call is
        evaluated, so the default behaviour is unchanged and correct.

        DO NOT use the cached path if you read obs["odor_intensity"] at a
        finer granularity than the exchange.
        """
        cd = self._countdown
        if cd > 1:
            self._countdown = cd - 1
            if self._cache is not None:
                return self._cache
        self._countdown = 0
        q = np.asarray(antennae_pos, dtype=float)
        shape = self._shape(self._displacement(q))              # (n,w)
        self._cache = self._peak.T @ shape                      # (k,w)
        return self._cache

    @property
    def odor_dimensions(self) -> int:
        return self.n_dims

    def calibrate_gain(self, r_ref=30.0, z_sensor=0.5):
        """Set self.gain so that the on-axis downwind intensity at r_ref mm
        equals r_ref**-2.

        WHY THIS MATTERS AND IS NOT COSMETIC: bridge.py's intensity_to_hz=4e4
        was chosen so that FlyGym's r^-2 field puts the fly mid-range rather
        than saturated ("~1e-3 near the source"). If we swap in a plume with a
        different absolute scale, that constant silently changes the ORN base
        rate and therefore the whole result, while looking like a pure
        "environment" change. Pinning the on-axis value at r_ref reproduces the
        old absolute scale exactly at that radius, so intensity_to_hz stays
        meaningful and the change is confined to the field's SHAPE.
        """
        g_was, i_was = self.gain, self.intermittency
        self.gain, self.intermittency = 1.0, 0.0
        try:
            # one probe, r_ref mm straight downwind of source 0, at sensor height
            d = np.zeros((1, 1, 3))
            d[0, 0, 0] = self._wind[0] * r_ref
            d[0, 0, 1] = self._wind[1] * r_ref
            d[0, 0, 2] = z_sensor - self._pos[0, 2]
            raw = float(self._shape(d)[0, 0])
        finally:
            self.gain, self.intermittency = g_was, i_was
        if raw <= 0.0:
            raise RuntimeError("calibration probe landed on a zero of the field")
        self.gain = (r_ref ** -2) / raw
        return self.gain

    def set_intermittency(self, amount, filament_mm=25.0):
        """Turbulent filaments, 0 = smooth plume, 1 = full dropout.

        FLAG: this is realistic and it will hurt. bridge.py's readout_tau_ms is
        already at 400 ms because the DNa02 contrast signal (~8.6 Hz per unit
        ORN contrast) barely clears its own Poisson shot noise. Filaments add a
        second, larger noise source at a timescale the EMA cannot reject. Turn
        it on only when you are deliberately studying intermittency, and expect
        the taxis result to degrade.
        """
        self.intermittency = float(amount)
        self.filament_mm = float(filament_mm)

    # ------------------------------------------------------------- live world
    def set_reserve(self, reserve):
        """Write new reserves and keep EVERY cached copy consistent.

        THE TRAP: OdorArena precomputes `_peak_intensity_repeated` in __init__
        (sensory_environment.py:143-147) and get_olfaction reads that, not
        `peak_odor_intensity`. So mutating `arena.peak_odor_intensity` alone
        does nothing at all -- silently. Our own get_olfaction reads
        `self._peak`, but we mirror into both parent attributes so that
        anything else that calls OdorArena.get_olfaction (or reads
        odor_dimensions) still sees the truth.
        """
        self.reserve = np.clip(np.asarray(reserve, dtype=float),
                               self.reserve_floor, 1.0)
        self._peak = self._peak0 * self.reserve[:, None]
        self.peak_odor_intensity = self._peak
        rep = self._peak[:, :, np.newaxis]
        self._peak_intensity_repeated = np.repeat(rep, self.num_sensors, axis=2)
        self._cache = None          # the field just changed; never serve stale

    def feed(self, fly_xy, dt_s):
        """Advance depletion / regrowth. Call from the loop, once per exchange.

        Returns the boolean (n,) "is feeding" mask.

        WHY NOT IN arena.step(): Simulation.step calls
        `self.arena.step(dt=self.timestep, physics=self.physics)` on EVERY
        physics step (simulation.py:212) -- 10,000 calls per simulated second.
        arena.step also never receives the fly's position. Driving depletion
        from the loop at the 200 Hz exchange rate is 50x cheaper and needs no
        reach-into-physics to find the fly.
        """
        d = self._pos[:, :2] - np.asarray(fly_xy, dtype=float)
        if self.period is not None:
            d = d - self.period * np.round(d / self.period)
        near = (d[:, 0] ** 2 + d[:, 1] ** 2) < self.feed_radius ** 2
        feeding = near & self.edible

        self.visits += (feeding & ~self._was_feeding).astype(np.int64)
        self._was_feeding = feeding
        self.dwell_s += feeding * dt_s

        r = self.reserve.copy()
        take = np.where(feeding, dt_s / self.deplete_tau_s, 0.0)
        eaten = np.minimum(take, np.maximum(r - self.reserve_floor, 0.0))
        self.consumed += eaten
        r = r - eaten
        if self.regrow_tau_s is not None:
            grow = (1.0 - r) * (dt_s / self.regrow_tau_s)
            r = r + np.where(feeding, 0.0, grow)
        self.set_reserve(r)
        return feeding

    def step(self, dt: float, physics=None, *args, **kwargs) -> None:
        """Called by Simulation.step every physics step. Keep it to one float
        add -- at 1e-4 s that is 10,000 calls per simulated second."""
        self.curr_time += dt

    def reset_world(self):
        self.set_reserve(np.ones(self.n_sources))
        self.consumed[:] = 0.0
        self.visits[:] = 0
        self.dwell_s[:] = 0.0
        self._was_feeding[:] = False
        self.curr_time = 0.0

    # ------------------------------------------------------------- utilities
    def field_on_grid(self, half, n=241, z=0.5, dim=0, center=(0.0, 0.0)):
        """(X, Y, I) for plotting the odour landscape of dimension `dim`."""
        xs = np.linspace(center[0] - half, center[0] + half, n)
        ys = np.linspace(center[1] - half, center[1] + half, n)
        X, Y = np.meshgrid(xs, ys)
        q = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], axis=1)
        d = self._displacement(q)
        shape = self._shape(d)                       # (n_src, n_pts)
        I = (self._peak[:, dim][:, None] * shape).sum(axis=0)
        return X, Y, I.reshape(X.shape)

    def wrap_xy(self, xy):
        """Absolute coordinates -> coordinates inside one unit cell."""
        xy = np.asarray(xy, dtype=float)
        if self.period is None:
            return xy
        return xy - self.period * np.floor(xy / self.period + 0.5)

    def _get_max_floor_height(self) -> float:
        return 0.0


# --------------------------------------------------------------------------- #
#  A world worth living in
# --------------------------------------------------------------------------- #
def make_world(period=300.0, wind=(1.0, 0.0), **kw) -> LivingWorldArena:
    """Two attractive patches of different strength plus one aversive patch.

    Geometry inside one 300 x 300 mm cell (the cell is centred on the origin,
    so coordinates run -150..150):

        FOOD_A  (+60, +55)   strong, depletes fast, regrows slowly
        FOOD_B  (-70, -45)   weaker, regrows faster
        BITTER  ( +5, -60)   aversive, never depletes, sits between them

    The bitter patch is deliberately NOT in a corner: it is placed roughly on
    the straight line from FOOD_B to FOOD_A, so a fly that simply climbs the
    attractive gradient has to deal with it rather than never meeting it.
    """
    sources = [
        Source(pos=(60.0, 55.0, 1.5),  intensity=(1.0, 0.0),
               rgba=(0.20, 0.72, 0.30, 1.0), edible=True,  label="FOOD_A"),
        Source(pos=(-70.0, -45.0, 1.5), intensity=(0.55, 0.0),
               rgba=(0.35, 0.85, 0.55, 1.0), edible=True,  label="FOOD_B"),
        Source(pos=(5.0, -60.0, 1.5),  intensity=(0.0, 0.8),
               rgba=(0.85, 0.25, 0.20, 1.0), edible=False, label="BITTER"),
    ]
    return LivingWorldArena(sources, period=period, wind=wind, **kw)


def world_camera_params(view_mm=140.0, fovy=45.0):
    """Tracking overhead camera whose frame is `view_mm` tall.

    The repo currently uses camera_name="camera_top", which the FlyGym config
    (flygym/config.yaml) defines as mode=track, pos=[0,0,8]. At fovy=45 that
    frame is 2*8*tan(22.5) = 6.6 mm tall: you see the fly and nothing else --
    no odour source, no landscape, no evidence of a world. For a persistent
    world raise the camera.

    Built by copying the known-good `camera_top_zoomout` entry (mode=track,
    pos=[0,0,40]) and overriding pos/fovy, so the `class: nmf` defaults and
    `ipd` that FlyGym's own cameras carry are preserved. Pass the result as
    Camera(camera_name="...", camera_parameters=...).
    """
    from flygym.util import load_config
    params = dict(load_config()["cameras"]["camera_top_zoomout"])
    params.pop("name", None)
    h = view_mm / (2.0 * np.tan(np.deg2rad(fovy) / 2.0))
    params["pos"] = [0.0, 0.0, float(h)]
    params["fovy"] = float(fovy)
    return params
