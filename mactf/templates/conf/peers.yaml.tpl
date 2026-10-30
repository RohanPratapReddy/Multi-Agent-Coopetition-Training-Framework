# conf/peers.yaml — owns: PEERS. Each entry is a class with kwargs; the class is a
# class name looked up in the app registry when it populates, not when this file is
# read. MLPPeer is defined in {{ project_name }}/peers/__init__.py.

PEERS:
  - name: peer_alpha
    class: MLPPeer
    kwargs:
      state_dim: 16
      action_dim: 4
      hidden: 64
  - name: peer_beta
    class: MLPPeer
    kwargs:
      state_dim: 16
      action_dim: 4
      hidden: 64
  - name: peer_gamma
    class: MLPPeer
    kwargs:
      state_dim: 16
      action_dim: 4
      hidden: 64
