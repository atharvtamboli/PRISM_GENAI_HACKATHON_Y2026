# Vision: Interruptible Agent — Full-Duplex Voice Assistant

**A full-duplex voice agent that can listen, respond, get interrupted, recover from stale intent, and safely execute actions.**

This project is a LiveKit-based Python voice agent built for **Full-Duplex-Bench v3**. The main focus is not just getting a voice response out quickly, but handling the messy parts of real conversations: interruptions, corrections, tool calls, and actions that may already be in progress.

It also includes **VisionGuide**, an end-to-end voice + vision extension that lets the agent query the user's current surroundings through a camera pipeline.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [How Full-Duplex / Interruption Handling Works](#how-full-duplex--interruption-handling-works)
- [Tool & Action Architecture](#tool--action-architecture)
- [Model / Provider Details](#model--provider-details)
- [FDB-v3 Benchmark](#fdb-v3-benchmark)
- [Benchmark Results](#benchmark-results)
- [Extension Use Case](#extension-use-case)
- [Installation](#installation)
- [Configuration / API Keys](#configuration--api-keys)
- [Running the Agent](#running-the-agent)
- [One-Command Reproduction](#one-command-reproduction)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [Demo Video](#demo-video)
- [Team / Credits](#team--credits)
- [References](#references)

---

## Project Overview

Most voice agents work like this:

```text
User speaks
    ↓
STT
    ↓
LLM
    ↓
TTS
    ↓
Agent speaks
```

That works for simple conversations, but real users do not always wait quietly for an agent to finish.

Our agent is built around a different idea:

```text
User
 │
 │ speaks
 ▼
┌───────────────────────────────────────┐
│        Full-Duplex Voice Agent        │
│                                       │
│  Listen ──► Reason ──► Act ──► Speak │
│      ▲             │          │       │
│      └── interrupt ┘          │       │
│                               │       │
│                 Tools ◄───────┘       │
└───────────────────────────────────────┘
```

The agent can continue working while the conversation changes, rather than treating every turn as an isolated request.

---

## Problem Statement

Traditional voice pipelines have a few practical problems:

- The user may interrupt while the agent is speaking.
- The user's latest statement may invalidate an action that was already being prepared.
- Tool calls can take time while the user continues talking.
- A stale response can be spoken after the user has already changed their request.
- Repeating or retrying an action can accidentally execute a mutation twice.

The goal of this project is to make voice interaction behave more like a real conversation:

> **The latest user intent should matter, stale work should be discarded, and state-changing actions should not accidentally happen twice.**

---

## Key Features

| Feature | Status |
| --- | --- |
| Real-time LiveKit voice agent | ✅ |
| Full-duplex interaction | ✅ |
| User interruption handling | ✅ |
| Async tool execution | ✅ |
| Stale-intent recovery | ✅ |
| Exactly-once mutation handling | ✅ |
| Function/tool calling | ✅ |
| Tool latency simulation | ✅ |
| Latency and tool-call tracking | ✅ |
| Self-correction handling | ✅ |
| Multi-step tool interactions | ✅ |
| VisionGuide extension | ✅ |
| YOLO11n scene detection | ✅ |
| Voice access to current scene | ✅ |
| Full-Duplex-Bench v3 evaluation | ✅ |

---

## Architecture

```text
                         ┌──────────────────┐
                         │       USER       │
                         │   Voice Input    │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │       VAD        │
                         │     Silero       │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │       STT        │
                         │  Deepgram Nova-3 │
                         └────────┬─────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │          LLM             │
                    │   GPT-4o-mini           │
                    │   LiveKit Inference      │
                    └───────────┬──────────────┘
                                │
                    ┌───────────┼───────────┐
                    │           │           │
                    ▼           ▼           ▼
               ┌────────┐ ┌──────────┐ ┌───────────┐
               │ Tools  │ │  Vision  │ │  Intent   │
               │ / APIs │ │  Guide   │ │  Recovery │
               └───┬────┘ └────┬─────┘ └─────┬─────┘
                   │            │             │
                   ▼            ▼             ▼
              Mock APIs     YOLO11n       Latest Intent
                   │        scene.json         │
                   └────────────┬─────────────┘
                                │
                                ▼
                         ┌──────────────┐
                         │     TTS      │
                         │ Cartesia     │
                         │  Sonic-3     │
                         └──────┬───────┘
                                │
                                ▼
                              USER
```

---

## How Full-Duplex / Interruption Handling Works

The important difference is that the agent does not treat the conversation as:

```text
WAIT → PROCESS → SPEAK → FINISH → LISTEN
```

Instead, processing can overlap with the conversation:

```text
User speaks
     │
     ├──────────────► STT
     │                  │
     │                  ▼
     │               LLM starts
     │                  │
     │                  ├──────► Tool call
     │                  │
User interrupts ────────┘
     │
     ▼
Latest intent becomes authoritative
     │
     ├── cancel / ignore stale work
     └── continue with updated request
```

### Interruption flow

1. User starts speaking.
2. VAD detects the new speech.
3. The agent can stop or invalidate the current response.
4. The new utterance is processed.
5. If the new request changes the previous intent, stale work is not allowed to become the final response.
6. The agent continues from the newest valid state.

This is especially important when a tool call is already running.

---

## Tool & Action Architecture

The agent uses tools for actions that should not be handled as plain LLM text.

```text
                    LLM
                     │
              chooses a tool
                     │
                     ▼
              ┌──────────────┐
              │ Action Layer │
              └──────┬───────┘
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       Travel     Finance    Housing
          │          │          │
          └──────────┼──────────┘
                     ▼
                E-commerce
```

The repository currently uses simulated APIs for travel, finance, housing, and e-commerce operations.

These mock APIs are useful for testing tool behaviour and latency without performing real transactions.

### Stale Intent

A tool result is not automatically trusted just because it finishes successfully.

The agent checks whether the result still belongs to the current user intent.

```text
Request A
   ↓
Tool A starts
   ↓
User says "Actually, do B"
   ↓
Intent changes
   ↓
Tool A result becomes stale
   ↓
Do not use stale result as the final action
```

### Exactly-Once Mutations

For state-changing actions, the goal is:

```text
One user action
      ↓
One committed mutation
```

rather than:

```text
retry → retry → duplicate action
```

This is important for any future real-world integration involving bookings, purchases, account changes, or other state-changing operations.

---

## Model / Provider Details

| Component | Provider / Model |
| --- | --- |
| Voice infrastructure | LiveKit Agents |
| VAD | Silero |
| STT | Deepgram Nova-3 |
| LLM | OpenAI GPT-4o-mini via LiveKit Inference |
| TTS | Cartesia Sonic-3 |
| Noise cancellation | LiveKit Background Noise Cancellation |
| Vision detection | Ultralytics YOLO11n |
| Camera processing | OpenCV |

The project does **not** depend on the older Gemma/Fish Audio configuration from the original starter setup.

---

## FDB-v3 Benchmark

The project was built and evaluated using **Full-Duplex-Bench v3**.

The benchmark is useful because normal chatbot testing does not fully capture what happens when users:

- interrupt the agent,
- correct themselves,
- change intent,
- trigger multiple tools,
- or continue speaking while an action is running.

The evaluation process was used to identify failures and iterate on:

- interruption handling,
- self-correction,
- tool chains,
- stale responses,
- and conversational latency.

---

## Benchmark Results

Current recorded benchmark run:

| Metric | Result |
| --- | ---: |
| Scenarios tested | **100** |
| Task completion | **88 / 100** |
| Recorded latency | **381 ms** |

These numbers are from the current benchmark run and should be treated as the result of this version of the agent, not as a universal performance guarantee.

The benchmark work also exposed the main remaining issues: interruption quality, fragmented STT, and failures in some complex multi-step interactions.

---

## Extension Use Case

### VisionGuide — Voice + Vision Assistant

Beyond the benchmark, the project was extended with **VisionGuide**.

VisionGuide adds a camera pipeline:

```text
Webcam
  ↓
OpenCV
  ↓
YOLO11n
  ↓
Object detection
  ↓
Position estimation
  ↓
Rough proximity
  ↓
scene.json
  ↓
get_current_scene()
  ↓
Voice Agent
```

The user can ask the voice agent about the current surroundings.

Example:

```text
User:
"What is around me?"

Agent:
"There's an object slightly to your left..."
```

VisionGuide can identify configured objects and estimate whether they are on the left, center, or right.

### Why this extension?

It gives the full-duplex architecture a real end-to-end use case where:

- the user talks naturally,
- the agent reasons about the request,
- the agent accesses a tool,
- the tool reads live visual context,
- and the result is returned through voice.

> VisionGuide is experimental and is not intended for safety-critical navigation.

---

## Installation

### Prerequisites

- Python
- LiveKit project
- LiveKit credentials
- Required provider API credentials
- Webcam if using VisionGuide

### Clone

```bash
git clone <your-repository-url>
cd interruptible-agent
```

### Install dependencies

Install the dependencies defined by `pyproject.toml`.

The voice-agent dependencies and the VisionGuide dependencies are separate concerns. VisionGuide uses OpenCV and Ultralytics, which are imported by the vision code.

---

## Configuration / API Keys

Create:

```text
.env
```

using:

```text
.env.example
```

as the reference.

The environment should contain the credentials required by the LiveKit and model providers used by the agent.

Do **not** commit `.env` or API keys.

Typical provider configuration includes:

```text
LIVEKIT_URL=...
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
```

Any additional provider credentials required by the current project should be added according to the existing `.env.example`.

---

## Running the Agent

The main entry point is:

```text
src/agent.py
```

Run the local agent with:

```bash
python src/agent.py dev
```

The agent connects to LiveKit and starts the voice session.

### Running VisionGuide

VisionGuide runs independently:

```text
vision_guide/vision.py
```

It updates:

```text
vision_guide/scene.json
```

The LiveKit agent reads the latest scene through the `get_current_scene` tool.

The camera process is **not automatically started by the voice-agent entry point**.

---

## One-Command Reproduction

After installing dependencies and configuring `.env`:

```bash
python src/agent.py dev
```

This starts the main voice-agent workflow.

For the complete benchmark reproduction, use the Full-Duplex-Bench v3 runner included in the repository and the benchmark data/configuration supplied with the project.

---

## Testing

The repository contains:

```text
tests/
scenarios.yaml
Full-Duplex-Bench/
```

Testing should cover:

- Basic voice conversations
- Interruptions while speaking
- User corrections
- Stale tool results
- Multi-step tool calls
- Tool latency
- Repeated state-changing requests
- Vision queries
- Benchmark scenarios

### Manual interruption test

Try:

```text
User:
"Book me a..."

Agent starts responding

User:
"Actually, don't book it. Just tell me the price."
```

The important behaviour is that the second instruction becomes the active intent and the earlier action should not continue into an unintended final mutation.

---

## Project Structure

```text
interruptible-agent/
│
├── src/
│   ├── agent.py              # Main LiveKit voice agent
│   ├── mock_apis.py          # Simulated service operations
│   └── latency_injector.py   # Tool latency simulation
│
├── vision_guide/
│   ├── vision.py             # Webcam + YOLO11n pipeline
│   ├── navigation.py         # Detection / guidance logic
│   ├── scene.json            # Latest scene snapshot
│   └── yolo11n.pt            # YOLO11n model
│
├── tests/                    # Tests
├── Full-Duplex-Bench/        # FDB-v3 benchmark
├── scenarios.yaml            # Conversation scenarios
│
├── pyproject.toml            # Python project configuration
├── Dockerfile                # Voice-agent container
├── .env.example              # Environment template
├── LICENSE
├── interruptible_agent.mp4   # Demo video
└── README.md
```

---

## Reproducibility

The repository keeps the main pieces needed to reproduce the project:

- Agent source code
- Tool implementations
- Vision pipeline
- Benchmark scenarios
- Configuration templates
- Docker configuration
- Evaluation resources
- Demo video

For a clean reproduction:

```text
1. Clone repository
2. Install dependencies
3. Configure .env
4. Start LiveKit agent
5. Run VisionGuide separately if needed
6. Run FDB-v3 evaluation
7. Compare results with the recorded benchmark run
```

The benchmark result reported above belongs to the tested version of the agent and may change with model/provider versions, network conditions, hardware, and configuration.

---

## Limitations

1. **Interruption is not perfect** — low-volume or poorly detected speech can still fail to interrupt the agent.
2. **STT fragmentation** — speech may sometimes be split into smaller chunks.
3. **Latency** — STT, LLM, tool execution, and TTS all contribute to end-to-end delay.
4. **Complex tool chains** — longer multi-step interactions can still fail.
5. **Mock APIs** — the current travel, finance, housing, and e-commerce tools do not perform real-world transactions.
6. **Vision accuracy** — YOLO performance depends on lighting, camera position, and the detected object.
7. **No true depth estimation** — VisionGuide uses a bounding-box-based proximity heuristic rather than physical distance.
8. **Separate camera process** — VisionGuide is not automatically started with the LiveKit agent.
9. **Safety** — VisionGuide should not be used as a safety-critical navigation system.
10. **Provider dependency** — the voice pipeline depends on external model/provider services.

---

## Demo Video

### Full Demo

[**▶ Watch `interruptible_agent.mp4`**](./interruptible_agent.mp4)

The demo covers the voice interaction, interruption behaviour, tool usage, and the extended VisionGuide use case.

---

## Team / Credits

**Project:** Interruptible Full-Duplex Voice Agent

**Track:** Samsung PRISM / Full-Duplex-Bench v3

Built using:

- LiveKit
- Deepgram
- OpenAI
- Cartesia
- Silero
- Ultralytics YOLO

Team members / contributors are listed in the repository commit history and project submission.

---

## References

- [LiveKit Agents](https://docs.livekit.io/agents/)
- [Full-Duplex-Bench](https://github.com/ServiceNow/Full-Duplex-Bench)
- [Deepgram](https://deepgram.com/)
- [OpenAI](https://openai.com/)
- [Cartesia](https://cartesia.ai/)
- [Ultralytics](https://docs.ultralytics.com/)

---

## License

See [`LICENSE`](LICENSE) for license details.
