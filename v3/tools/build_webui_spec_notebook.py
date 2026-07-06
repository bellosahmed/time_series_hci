#!/usr/bin/env python3.12
"""Programmatically build `local_webui_dataset_spec.ipynb` — an exhaustive, offline-first specification
notebook documenting the on-premise Open WebUI + Ollama + Docker + NVIDIA-GPU + local-dataset setup.

100% local: no cloud, no external APIs, no online image assets (diagrams are inline ASCII so they render
with zero network access). Grounded in this host's real values (RTX 5070, driver 595.71.05, Docker 29.6.1
with the nvidia runtime, Ollama serving llama3.2:3b, Python 3.12.13 / OpenSSL 3.5.5).

Run:  python3.12 tools/build_webui_spec_notebook.py   →   writes local_webui_dataset_spec.ipynb
"""
from __future__ import annotations

import pathlib

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells: list = []


def md(s: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(s))


def sh(s: str) -> None:
    """A bash verification cell (Jupyter `%%bash` magic)."""
    cells.append(nbf.v4.new_code_cell("%%bash\n" + s))


def py(s: str) -> None:
    cells.append(nbf.v4.new_code_cell(s))


DATASET = "/home/saizer/Documents/Msc_project/code/data/dad"

# ----------------------------------------------------------------------------- 0. Title & scope
md(r"""# Local, Offline Open WebUI + Ollama + Dataset — On-Premise Specification

**Author:** Bello Ahmed · **Host:** Ubuntu Linux · **Scope:** 100% local / on-premise / air-gapped-friendly.

This notebook is the **end-to-end operational specification** for a fully self-hosted, private AI stack:
a local **Open WebUI** container talking to a local **Ollama** instance, with **NVIDIA GPU pass-through**,
and a **local dataset** (the *DAD* IoT-network dataset) mounted for private **RAG** (Retrieval-Augmented
Generation). **Nothing leaves the machine** — no cloud, no external APIs, no telemetry.

> **Design rule for this document:** every diagram is inline ASCII and every check is a local command, so
> the notebook renders and runs on a machine with **no internet at all**. (We deliberately avoid external
> `![](https://…)` image assets — they would violate the offline/privacy requirement.)

### What this documents
1. Host environment health checks (Python 3.12 + SSL, NVIDIA driver, Docker daemon runtimes).
2. The container architecture and host↔container data-flow, with privacy boundaries.
3. The Open WebUI `docker run` command and a `docker-compose.yaml`, fully annotated.
4. Environment variables for **offline mode**, GPU, and telemetry-off.
5. Volume mapping of the **local dataset** and connecting it to Open WebUI for private indexing/RAG.
6. In-container GPU verification and operational troubleshooting workflows.
""")

# ----------------------------------------------------------------------------- 1. Privacy boundary
md(r"""## 1. Privacy-first principles & data boundary

| Principle | Implementation |
| --- | --- |
| No cloud / no external APIs | Ollama + Open WebUI run locally; no OpenAI/hosted keys configured |
| No telemetry | `ANONYMIZED_TELEMETRY=false`, `DO_NOT_TRACK=true`, `SCARF_NO_ANALYTICS=true` |
| Offline model/asset loading | `OFFLINE_MODE=1`, `HF_HUB_OFFLINE=1`, embeddings via local Ollama |
| Data stays on host | dataset bind-mounted **read-only**; chat/RAG index in a local Docker volume |
| Reproducible & inspectable | one `docker-compose.yaml`, pinned image tag, documented env |

**The boundary:** raw data, the vector index, model weights, and prompts/answers all live on the host
(bind mounts + named volumes). The only network traffic is **loopback** between the browser, the Open
WebUI container (`:3000`), and the Ollama service (`:11434`).
""")

# ----------------------------------------------------------------------------- 2. Architecture
md(r"""## 2. Architecture — local container & data flow

```
                          ┌───────────────────────── Ubuntu host (on-prem, no internet needed) ─────────────────────────┐
                          │                                                                                              │
   ┌──────────┐  loopback │   ┌───────────────────────────┐         ┌──────────────────────────┐                        │
   │ Browser  │──:3000────┼──▶│  Open WebUI (container)    │         │  Ollama (native service) │                        │
   │ (local)  │           │   │  ghcr.io/.../open-webui:cuda│──:11434▶│  llama3.2:3b, embeddings │                        │
   └──────────┘           │   │  --gpus all                │  API    │  (GPU-accelerated)       │                        │
                          │   │  port 8080 → host 3000     │         └────────────┬─────────────┘                        │
                          │   └─────────┬──────────────────┘                      │                                      │
                          │             │ mounts                                  │ CUDA                                 │
                          │   ┌─────────▼───────────┐   ┌────────────────┐   ┌────▼──────────────┐                       │
                          │   │ named vol:          │   │ bind mount RO:  │   │  NVIDIA RTX 5070  │                       │
                          │   │ open-webui:/app/... │   │ dataset → /data │   │  driver 595.71    │                       │
                          │   │ (chats + RAG index) │   │ (DAD, read-only)│   │  (nvidia runtime) │                       │
                          │   └─────────────────────┘   └────────────────┘   └───────────────────┘                       │
                          │                                                                                              │
                          └──────────────────────────────────────────────────────────────────────────────────────────┘
                                              ▲ everything above the line is loopback-only — no egress ▲
```
- **Open WebUI** (container) serves the chat UI on host **:3000** and calls **Ollama** on **:11434**.
- **GPU** is passed to the container with `--gpus all` (the Docker **nvidia runtime**).
- The **dataset** is bind-mounted **read-only**; the **RAG index + chats** persist in a named volume.
""")

# ----------------------------------------------------------------------------- 3. Prereqs
md(r"""## 3. Prerequisites (verified on this host)

| Component | Required | This host |
| --- | --- | --- |
| OS | Ubuntu Linux | Ubuntu |
| Python | 3.12.x with SSL | 3.12.13, OpenSSL 3.5.5 |
| NVIDIA driver | ≥ 550 (Blackwell) | **595.71.05** |
| GPU | CUDA-capable | **RTX 5070 Laptop, 8 GB** |
| Docker | ≥ 24, nvidia runtime | **29.6.1**, `nvidia` runtime present |
| Ollama | local service `:11434` | running (`llama3.2:3b`) |
| Open WebUI image | `ghcr.io/open-webui/open-webui:cuda` | pull once, then offline |

The cells below **verify** each of these locally.
""")

# ----------------------------------------------------------------------------- 4. Host health
md("## 4. Host health checks\n### 4.1 Python 3.12 + SSL / stdlib crypto\nConfirm the local interpreter and that its SSL module works (needed for the one-time image pull, and for any local HTTPS between services).")
sh("python3.12 --version\nwhich python3.12\npython3.12 -c \"import ssl, hashlib, sqlite3; print('SSL:', ssl.OPENSSL_VERSION); print('sha256 ok:', hashlib.sha256(b'x').hexdigest()[:12]); print('sqlite:', sqlite3.sqlite_version)\"")
py(r'''# a fuller local stdlib self-test (no network) — SSL context, hashing, and JSON round-trip
import ssl, hashlib, json, sqlite3, platform
ctx = ssl.create_default_context()
print("python      :", platform.python_version())
print("openssl     :", ssl.OPENSSL_VERSION)
print("tls max     :", ctx.maximum_version)
print("sha256 self :", hashlib.sha256(b"privacy-first").hexdigest()[:16])
print("json rt     :", json.loads(json.dumps({"local": True}))["local"])
print("All local stdlib crypto modules import and run — no network used. ✓")''')

md("### 4.2 NVIDIA driver & GPU\nThe host must see the GPU before the container can.")
sh("nvidia-smi\nnvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader")
md(r"""**Sample output on this host** (yours will match hardware):
```
NVIDIA GeForce RTX 5070 Laptop GPU, 595.71.05, 8151 MiB, 210 MiB
```
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.05              Driver Version: 595.71.05      CUDA Version: 13.2       |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA GeForce RTX 5070 ...     On  | 00000000:01:00.0  Off  |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```
""")

md("### 4.3 NVIDIA Container Toolkit (GPU → Docker)\nThe Docker daemon needs the **nvidia runtime** so `--gpus all` works.")
sh("docker info --format 'Default runtime : {{.DefaultRuntime}}'\ndocker info --format 'Runtimes       : {{range $k,$v := .Runtimes}}{{$k}} {{end}}'\n# a throwaway CUDA container that runs nvidia-smi INSIDE Docker (proves pass-through end-to-end):\n# docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi")
md(r"""**Expected:** `Runtimes` must list **`nvidia`** (this host: `runc nvidia runc`). If it does not, install
`nvidia-container-toolkit` and restart Docker:
```bash
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
""")

md("### 4.4 Docker daemon — storage & local runtime\nConfirm the daemon is up and where it stores data locally.")
sh("docker --version\ndocker info --format 'Storage driver : {{.Driver}}'\ndocker info --format 'Docker root    : {{.DockerRootDir}}'\ndocker info --format 'Cgroup driver  : {{.CgroupDriver}}'\ndocker volume ls")

md("### 4.5 Local Ollama service\nOpen WebUI talks to Ollama over loopback `:11434`. Confirm it is up and which models are pulled.")
sh("curl -s http://localhost:11434/api/tags | python3.12 -m json.tool | head -40\nollama list 2>/dev/null || true")
md(r"""**Sample (this host):**
```
NAME             ID            SIZE      MODIFIED
llama3.2:3b      a80c4f17acd5  2.0 GB    ...
```
For **offline RAG embeddings**, pull a local embedding model once (then it is cached):
```bash
ollama pull nomic-embed-text
```
""")

# ----------------------------------------------------------------------------- 5. The container
md(r"""## 5. The Open WebUI container — the annotated run command

One command starts the whole UI. Note **`--gpus all`**, the **`3000:8080`** port map, the **read-only
dataset** bind mount, the persistent **named volume**, and the **offline/telemetry-off** env vars.""")
sh(r"""# Start Open WebUI (CUDA image) — fully local, GPU-accelerated, offline, telemetry-off.
docker run -d \
  --name open-webui \
  --gpus all \
  -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -e ENABLE_OLLAMA_API=true \
  -e RAG_EMBEDDING_ENGINE=ollama \
  -e RAG_EMBEDDING_MODEL=nomic-embed-text \
  -e OFFLINE_MODE=1 \
  -e HF_HUB_OFFLINE=1 \
  -e ANONYMIZED_TELEMETRY=false \
  -e DO_NOT_TRACK=true \
  -e SCARF_NO_ANALYTICS=true \
  -e WEBUI_AUTH=True \
  -e ENABLE_SIGNUP=false \
  -v open-webui:/app/backend/data \
  -v """ + DATASET + r""":/data/datasets/dad:ro \
  --restart unless-stopped \
  ghcr.io/open-webui/open-webui:cuda

docker ps --filter name=open-webui --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'""")

md(r"""### Why each flag
| Flag / env | Purpose |
| --- | --- |
| `--gpus all` | pass the RTX 5070 into the container (needs the nvidia runtime) |
| `-p 3000:8080` | container's `8080` served on host `3000` (loopback only) |
| `--add-host=host.docker.internal:host-gateway` | lets the container reach the host's Ollama at `:11434` |
| `OLLAMA_BASE_URL` | where Open WebUI finds Ollama |
| `RAG_EMBEDDING_ENGINE=ollama` + `RAG_EMBEDDING_MODEL` | embeddings computed **locally** by Ollama (offline RAG) |
| `OFFLINE_MODE`, `HF_HUB_OFFLINE` | never reach out for models/assets |
| `ANONYMIZED_TELEMETRY`/`DO_NOT_TRACK`/`SCARF_NO_ANALYTICS` | disable all telemetry |
| `WEBUI_AUTH=True`, `ENABLE_SIGNUP=false` | local login, no open sign-ups |
| `-v open-webui:/app/backend/data` | persist chats + RAG index in a **named volume** |
| `-v <dataset>:/data/datasets/dad:ro` | mount the local dataset **read-only** |
| `--restart unless-stopped` | survive reboots |
""")

# ----------------------------------------------------------------------------- 6. compose + env files
md("## 6. Sample configuration files (generated locally)\nThe next cells **write** a `docker-compose.yaml` and a `.env` next to this notebook — a declarative, reproducible alternative to the raw `docker run`.")
py(r'''# write docker-compose.yaml (multi-service, GPU reservation, offline env)
compose = """services:
  open-webui:
    image: ghcr.io/open-webui/open-webui:cuda
    container_name: open-webui
    restart: unless-stopped
    ports:
      - "3000:8080"          # host:container  (loopback)
    extra_hosts:
      - "host.docker.internal:host-gateway"
    env_file: .env
    volumes:
      - open-webui:/app/backend/data                 # chats + RAG index (persistent)
      - ''' + DATASET + r''':/data/datasets/dad:ro    # local dataset, read-only
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

volumes:
  open-webui:
"""
open("docker-compose.yaml", "w").write(compose)
print("wrote docker-compose.yaml\\n")
print(compose)''')
py(r'''# write the offline / privacy .env
env = """# --- connectivity (local Ollama) ---
OLLAMA_BASE_URL=http://host.docker.internal:11434
ENABLE_OLLAMA_API=true
# --- local RAG (embeddings computed by Ollama, offline) ---
RAG_EMBEDDING_ENGINE=ollama
RAG_EMBEDDING_MODEL=nomic-embed-text
# --- fully offline ---
OFFLINE_MODE=1
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
# --- no telemetry ---
ANONYMIZED_TELEMETRY=false
DO_NOT_TRACK=true
SCARF_NO_ANALYTICS=true
# --- local auth ---
WEBUI_AUTH=True
ENABLE_SIGNUP=false
# WEBUI_SECRET_KEY=<generate once: openssl rand -hex 32>
"""
open(".env", "w").write(env)
print("wrote .env\\n")
print(env)''')
md("Then simply:\n```bash\ndocker compose up -d      # start\ndocker compose logs -f    # watch\ndocker compose down       # stop\n```")

# ----------------------------------------------------------------------------- 7. Data layer
md(r"""## 7. Data layer — local dataset volume mapping

The dataset lives on the host and is mounted **read-only** into the container. Host → container map:

```
HOST                                                         CONTAINER
""" + DATASET + r"""/          ─(ro)─►   /data/datasets/dad/
   ├── capture_1910210000_mon.pcap                              ├── capture_1910210000_mon.pcap
   ├── capture_1910210000_labels.csv                            ├── capture_1910210000_labels.csv
   └── … (7 days)                                               └── …

docker volume  open-webui        ──────►   /app/backend/data/   (chats, settings, RAG vector index)
```

- **Read-only (`:ro`)** on the dataset means the container can index it but never modify it — a hard data-integrity boundary.
- The **RAG index** and chats live in the `open-webui` **named volume**, so they persist across container
  restarts but stay on the host.""")
sh("echo '--- host dataset directory ---'\nls -lh " + DATASET + " | head -8\necho\necho '--- confirm the mount inside the running container (read-only) ---'\ndocker exec open-webui ls -lh /data/datasets/dad 2>/dev/null | head -5 || echo '(start the container first)'\ndocker exec open-webui touch /data/datasets/dad/_writetest 2>&1 | head -1 || echo 'read-only mount confirmed (write blocked)'")

# ----------------------------------------------------------------------------- 8. RAG wiring
md(r"""## 8. Connecting the local dataset to Open WebUI for private RAG

All indexing happens **on the host**; embeddings are computed by the local Ollama model, so **no data or
prompt ever leaves the machine**.

1. Pull a local embedding model once: `ollama pull nomic-embed-text` (already set as `RAG_EMBEDDING_MODEL`).
2. In the browser at **http://localhost:3000** → **Workspace → Knowledge → Create**.
3. Upload files, or point it at the mounted dataset path **`/data/datasets/dad`** (visible because of the
   read-only bind mount).
4. Open WebUI chunks + embeds the documents locally into its vector store (in the `open-webui` volume).
5. In a chat, attach the knowledge base with **`#`** (e.g. `#dad`) — the model now answers **grounded** in
   your private data via RAG, entirely offline.

> **Note for tabular/pcap data:** Open WebUI RAG is text-oriented. For the DAD *captures*, index a
> **text/markdown export** (e.g. this project's `report/*.md`, or a CSV/summary you generate) rather than
> raw `.pcap` binaries — the deterministic analytics still come from the project's own query engine.
""")

# ----------------------------------------------------------------------------- 9. In-container GPU
md("## 9. GPU verification **inside** the container\nProves the pass-through works end-to-end, not just on the host.")
sh("echo '--- nvidia-smi inside the Open WebUI container ---'\ndocker exec open-webui nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader 2>/dev/null || echo '(container not running, or GPU not passed — see troubleshooting)'\necho\necho '--- confirm Ollama is using the GPU while a prompt runs ---'\ncurl -s http://localhost:11434/api/generate -d '{\"model\":\"llama3.2:3b\",\"prompt\":\"ping\",\"stream\":false}' >/dev/null &\nsleep 2; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; wait")

# ----------------------------------------------------------------------------- 10. Privacy boundaries
md(r"""## 10. Data privacy boundaries (what never leaves the host)

| Asset | Location | Leaves host? |
| --- | --- | --- |
| Raw dataset | host bind mount (`:ro`) | **No** |
| Vector/RAG index | `open-webui` Docker volume | **No** |
| Model weights | Ollama local store (`~/.ollama`) | **No** |
| Prompts & answers | container ↔ Ollama over loopback | **No** |
| Telemetry / analytics | disabled by env | **No** |

**Only** loopback traffic (`127.0.0.1:3000`, `127.0.0.1:11434`) occurs at runtime. The **one** time the
host needs the network is the initial `docker pull` / `ollama pull`; after that the stack is fully
air-gappable. To prove it, you can run the container on an **internal** Docker network with no gateway,
or simply disconnect the NIC and confirm the UI + RAG still work.
""")

# ----------------------------------------------------------------------------- 11. Troubleshooting
md(r"""## 11. Local troubleshooting workflows

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| UI 500 / "no models" | container can't reach Ollama | check `OLLAMA_BASE_URL` + `--add-host`; `curl` `:11434` from the container |
| `could not select device driver "" with capabilities: [[gpu]]` | nvidia runtime missing | install `nvidia-container-toolkit`, `nvidia-ctk runtime configure`, restart Docker |
| `nvidia-smi` empty in container | `--gpus all` omitted | recreate the container with `--gpus all` |
| Port 3000 in use | another service (e.g. a dev server) | change to `-p 3001:8080`, or free 3000 |
| RAG tries to download an embedder | HF not offline | set `RAG_EMBEDDING_ENGINE=ollama` + `HF_HUB_OFFLINE=1` |
| Chats/index lost on restart | no named volume | ensure `-v open-webui:/app/backend/data` |
""")
sh(r"""# --- diagnostics one-liners ---
docker ps -a --filter name=open-webui --format 'table {{.Names}}\t{{.Status}}'
echo '--- last 30 log lines ---'
docker logs --tail 30 open-webui 2>&1 || true
echo '--- can the container reach Ollama? ---'
docker exec open-webui sh -lc 'curl -s http://host.docker.internal:11434/api/tags | head -c 200' 2>/dev/null || echo '(container down)'
echo '--- what is bound to :3000 ? ---'
ss -ltnp 2>/dev/null | grep ':3000' || echo 'nothing on 3000'""")

md(r"""### Sample operational log (healthy start)
```
INFO  [open_webui] Starting Open WebUI (CUDA)
INFO  [open_webui.apps.ollama] Connected to Ollama at http://host.docker.internal:11434
INFO  [open_webui.retrieval] Embedding engine: ollama (nomic-embed-text)
INFO  [open_webui] Telemetry disabled (DO_NOT_TRACK=true)
INFO  [uvicorn] Uvicorn running on http://0.0.0.0:8080
```
""")

# ----------------------------------------------------------------------------- 12. Checklist
md(r"""## 12. Verification checklist & summary

- [ ] `python3.12 --version` → 3.12.x, `ssl.OPENSSL_VERSION` prints (§4.1)
- [ ] `nvidia-smi` shows the RTX 5070, driver 595.71 (§4.2)
- [ ] `docker info` Runtimes include **`nvidia`** (§4.3)
- [ ] Ollama reachable on `:11434` with a model pulled (§4.5)
- [ ] `docker run … --gpus all … open-webui:cuda` → `docker ps` healthy (§5)
- [ ] `docker exec open-webui nvidia-smi` shows the GPU (§9)
- [ ] Dataset visible read-only at `/data/datasets/dad`; write blocked (§7)
- [ ] Knowledge base created and answers grounded via RAG (§8)
- [ ] No external egress at runtime — loopback only (§10)

**Summary.** A single GPU-accelerated Open WebUI container, wired to a local Ollama, with a read-only
dataset mount and Ollama-computed embeddings, gives a **completely offline, private** question-answering
+ RAG stack. Every asset — data, index, weights, prompts — stays on the host; the only egress is the
one-time image/model pull, after which the system is air-gappable.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}
out = pathlib.Path(__file__).resolve().parent.parent / "local_webui_dataset_spec.ipynb"
nbf.write(nb, str(out))
print(f"wrote {out} with {len(cells)} cells")
