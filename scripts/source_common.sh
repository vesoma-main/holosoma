if [ -z "$WORKSPACE_DIR" ]; then
  WORKSPACE_DIR=$HOME/.holosoma_deps
fi
CONDA_ROOT=$WORKSPACE_DIR/miniconda3
export CONDA_ENVS_DIRS=$CONDA_ROOT/envs
