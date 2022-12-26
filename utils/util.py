from hydra import compose, initialize

initialize(version_base="1.3", config_path="../", job_name="ad2s")
cfg = compose(config_name="config", overrides=[])
