_base_="../base-pascal-config.py"

# this will merge with the parent

# epoch related
total_iters=400000
checkpoint_config = dict(interval=total_iters)
checkpoint_config = dict(interval=total_iters//4)
