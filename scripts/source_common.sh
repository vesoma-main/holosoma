if [ -z "$WORKSPACE_DIR" ]; then
  if [ -f /.dockerenv ]; then
    WORKSPACE_DIR=/workspace/.holosoma_deps
    # Redirect conda envs from /venv (ephemeral) to /workspace/venv (persistent)
    if [ ! -L /venv ]; then
      mkdir -p /workspace/venv
      if [ -d /venv ]; then
        cp -a /venv/. /workspace/venv/
        rm -rf /venv
      fi
      ln -s /workspace/venv /venv
    fi
  else
    WORKSPACE_DIR=$HOME/.holosoma_deps
  fi
fi
CONDA_ROOT=$WORKSPACE_DIR/miniconda3
