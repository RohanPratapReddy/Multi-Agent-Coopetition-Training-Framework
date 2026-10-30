# conf/rl.yaml — owns: RL. One key per function family, plus the switches. Omitting a
# family keeps the framework's behaviour for it; omitting the file entirely is the same
# as writing every value below.

RL:
  # Each family takes a name, {type, params} or {import, params} (section 5.1).
  # ADVANTAGE: gae
  # VALUE:     {type: mlp, params: {hidden: 64}}
  # LOSS:      {type: weighted_sum, params: {terms: [...], weights: [...]}}
  # ROLLUP:    kappa_weighted
  # SYNTHESIS: consensus

  # A per-node family: FAMILY -> glob -> spec. Not allowed for ROLLUP (E042: one rollup
  # for the whole tree).
  # OVERRIDES:
  #   ADVANTAGE:
  #     "peer-b*": {type: clipped, params: {clip: 1.0}}

  # A configured family no active consumer reads is an error (E041); "warn" is the
  # migration setting and reports W010.
  UNCONSUMED: error
  # An override glob that matches no node at build is an error (E044).
  STRICT_GLOBS: true
  # Run the contract probe on custom functions only ("custom"), on all ("all"), or
  # not at all ("off").
  VERIFY: custom
