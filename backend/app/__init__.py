import os

# Must be set before `import litellm` anywhere: use the cost map bundled with the package instead
# of downloading it at startup, so the service boots fast and works without outbound access.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
