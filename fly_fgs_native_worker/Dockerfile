FROM node:22.22.1-bookworm-slim@sha256:4f77a690f2f8946ab16fe1e791a3ac0667ae1c3575c3e4d0d4589e9ed5bfaf3d AS fly_fgs_node_runtime

FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7

ARG FLYGYM_ASSET_CACHE=/opt/flygym-assets
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MUJOCO_GL=egl \
    FLYGYM_ASSET_CACHE_DIR=${FLYGYM_ASSET_CACHE}

RUN apt-get update \
    && apt-get install --no-install-recommends -y libegl1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# The SHA-locked upstream circuit remains JavaScript authority. Copy only the
# exact Node executable needed by its RPC sidecar; npm and mutable packages are
# intentionally absent from the scientific runtime image.
COPY --from=fly_fgs_node_runtime /usr/local/bin/node /usr/local/bin/node
RUN test "$(node --version)" = "v22.22.1"

WORKDIR /app
COPY requirements.lock ./

RUN python -m pip install --no-cache-dir --require-hashes -r requirements.lock

# Keep the expensive pinned native dependency layer reusable when simulator
# source, benchmark registries, or tests change. The final layer copies the
# complete inventoried bundle so the in-container verifier checks the same
# content-addressed source that was transferred through Git.
COPY . ./

ENTRYPOINT ["python", "-m", "fly_sensor2behavior.cli"]
CMD ["--help"]
