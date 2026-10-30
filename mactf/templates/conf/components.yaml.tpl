# conf/components.yaml — owns: COMPONENTS and COMPLIANCE. A run that writes neither
# keyword behaves exactly as the framework always has: five strategies, six weight
# rules, no plugin, no import, and a restore difference reported as a warning.

COMPONENTS:
  # Replacing a built-in (``penalty``, ``kappa_softmax``, ``AdamW``, ...) needs this.
  # Off by default: a built-in is the framework's promise about a number.
  ALLOW_BUILTIN_OVERRIDE: false

  # Patterns, ``fnmatch``, two forms:
  #   "kind:name"       one component: cannot be registered over or removed
  #   "slot:<path>"     one slot: cannot be swapped or queued at run time
  # A production deployment pins the compliance boundary and freezes the rest.
  LOCKED: []
  #  - "scorer:penalty"
  #  - "slot:SCORING.CHAIN.penalty"
  #  - "slot:TRAINER.WEIGHT_UPDATE"

  # An old-style ``PEER_STRATEGIES["x"] = Cls`` write warns (W005). ``true`` makes it
  # an error, which is what a project that has finished migrating wants.
  STRICT_LEGACY: false

  # ``import: "pkg.mod:Class"`` in a spec executes a module. Trusted surfaces (conf/,
  # --set, code) may always; an HTTP request body may not unless ALLOW_IMPORT is on
  # *and* the module matches IMPORT_ALLOWLIST.
  ALLOW_IMPORT: false
  IMPORT_ALLOWLIST: []

  MAX_PER_KIND: 512

  # "auto" loads every installed ``mactf.components`` entry point; "off" loads none;
  # a list loads exactly those names.
  PLUGINS: auto

  # Dotted modules imported once, in order, before any spec is resolved. A module
  # named here is the project's opt-in (I5): registration is never a side effect of
  # being importable.
  MODULES: []

  # Refuse a new global component after startup. A run may still bring its own into
  # its overlay; a LOCKED slot may not change at all.
  FREEZE_AFTER_LOAD: false

  # A resume whose components differ from the checkpoint: false reports W052 and
  # continues with the configured component, true refuses with E094.
  STRICT_RESTORE: false

COMPLIANCE:
  # Lambda floor for every non-offsettable rule; > 0. A zero would erase a penalty.
  MIN_FLOOR_LAMBDA: 1.0
  # enforce: a guard violation fails the run. fallback: the reference stage scores
  # instead and W003 is logged. Never a silent clamp.
  GUARD_MODE: enforce
  # The largest reward total a custom reward stage may report.
  REWARD_CEILING: 10000.0
  # Floating-point slack in the guard's comparisons, in [0, 1e-3].
  TOLERANCE: 1.0e-9
  # Skip inspecting the three built-in stage classes (I1): they cannot violate.
  TRUST_BUILTIN: true
