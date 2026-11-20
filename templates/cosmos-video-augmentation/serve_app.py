"""Entry point for ``serve run serve_app:app`` and for the Anyscale Service in jobs/service.yaml.

USE_GPUS=1 runs Cosmos Transfer 2.5 live on GPU replicas (needs the template image, the
model weights and prepared clips). Unset, the same application runs on CPU in replay mode.
"""
import os

from offroad_ray.serve.app import build_app

app = build_app(use_gpus=os.environ.get("USE_GPUS", "0") == "1",
                replicas=int(os.environ.get("PAINT_REPLICAS", "8")),
                guardrails=os.environ.get("COSMOS_GUARDRAILS", "1") == "1")
