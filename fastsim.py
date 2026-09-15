"""Bit-exact speedups for the body half.

ONE change, and a deliberate refusal.

ADOPTED -- memoise get_observation within a timestep.
  HybridTurningController.step() calls super().get_observation() at its top (to
  feed the retraction/stumbling rules), and then super().step() -> Fly.post_step()
  builds the whole observation dict AGAIN. Between those two calls the physics
  state does not advance: post_step's obs is built after physics.step(), and the
  next step's top-of-loop obs is read before arena.step()/pre_step()/physics.step()
  run. Same qpos, same qvel, same sensordata -> same dict.
  Measured 0.507 ms per build, so this returns ~0.5 ms per timestep = ~25 ms per
  50-step exchange.

  Subtlety that makes it safe: dm_control marks physics dirty when Fly.pre_step
  writes ctrl through a Binding, and the first binding read inside post_step's
  get_observation triggers a full physics.forward(). That forward happens BEFORE
  post_step's dict is built, and it clears the dirty flag -- so by the time the
  next step reads the cache, no forward is pending and the state genuinely is
  identical. Guarded anyway: the cache is keyed on (curr_time, dirty-flag) and
  falls through to a real rebuild on any mismatch.

REFUSED -- deduplicating the collision pairs.
  Fly._init_self_contacts de-duplicates on the ORDERED key f"{geom1}_{geom2}", so
  both (A,B) and (B,A) are registered: 2,268 explicit pairs of which only 1,182
  are unique. Removing the 1,086 duplicates would cut broad-phase work, but
  measured over 3,000 walking steps NO duplicated pair was ever in contact (all
  11 contacts were leg-ground on unique pairs), so the saving is unproven for
  this gait -- and if legs ever do touch, removing a duplicate would change the
  contact force. That is a dynamics change, not an optimisation. Left alone.
  Worth reporting upstream to NeLy-EPFL.
"""
from __future__ import annotations

__all__ = ["memoise_observation", "observation_stats"]

_STATS = {"builds": 0, "hits": 0}


def observation_stats():
    return dict(_STATS)


def memoise_observation(sim):
    """Patch so get_observation() is built at most once per physics state.

    MUST PATCH THE CLASS, NOT THE INSTANCE. HybridTurningController.step calls
    `super().get_observation()`, and super() resolves on the class along the MRO
    -- it never consults the instance __dict__. An instance-level patch
    (sim.get_observation = ...) is therefore a silent no-op: measured 0 builds,
    0 hits, 0.99x. This patches the class that actually defines the method and
    keeps the cache per-instance, so concurrent Simulations cannot cross-talk.

    Returns sim. Idempotent.
    """
    # Patch Fly.get_observation, NOT Simulation.get_observation. Both calls in a
    # timestep bottom out in the Fly method -- the controller's top-of-step call
    # goes Simulation.get_observation -> fly.get_observation(sim), and post_step
    # calls fly.get_observation(sim) directly. Patching the Simulation method
    # intercepts only the first of the two: measured 1500 builds, 0 hits.
    # Fly.get_observation is also the 0.507 ms that the profile attributes.
    from flygym.fly import Fly
    owner = Fly
    if "get_observation" not in owner.__dict__:
        raise RuntimeError("Fly does not define get_observation")

    if not getattr(owner, "_obs_memoised", False):
        original = owner.__dict__["get_observation"]

        def _key(sim):
            # curr_time advances only in Simulation.step, and the dirty flag
            # flips when ctrl is written, so the pair identifies a physics state.
            try:
                dirty = bool(sim.physics.is_dirty)
            except Exception:
                dirty = False
            return (sim.curr_time, dirty)

        def cached(self, sim, *a, **kw):
            key = _key(sim)
            cache = getattr(self, "_obs_cache", None)
            if cache is not None and cache[0] == key:
                _STATS["hits"] += 1
                return cache[1]
            obs = original(self, sim, *a, **kw)
            _STATS["builds"] += 1
            # Re-key after building: the build itself may have cleared the dirty
            # flag via dm_control's lazy forward().
            self._obs_cache = (_key(sim), obs)
            return obs

        cached.__name__ = "get_observation"
        setattr(owner, "get_observation", cached)
        owner._obs_memoised = True
        owner._obs_original = original

    for fly in getattr(sim, "flies", []):
        fly._obs_cache = None
    return sim


def unmemoise_observation(sim=None):
    """Undo memoise_observation, for A/B testing in the same process."""
    from flygym.fly import Fly
    if getattr(Fly, "_obs_memoised", False):
        setattr(Fly, "get_observation", Fly._obs_original)
        del Fly._obs_memoised, Fly._obs_original
    for fly in getattr(sim, "flies", []) if sim is not None else []:
        fly._obs_cache = None
    return sim
