_base_="../base-bdd_hires-config.py"

# this will merge with the parent
model=dict(pretrained='data/basetrain_chkpts/moco_v2_800ep.pth')

# epoch related
total_iters=100000
checkpoint_config = dict(interval=total_iters)
