# Architecture

## Goal

Really-unREAL models observable conversational behavior from supplied history and lets a simulation continue from that history while real time passes.

The key architectural decision is to split **whether an action happens** from **what text is generated**.

```text
                         REAL CLOCK
                             |
                             v
                    Temporal Engine
                             |
                             v
                       Action Policy
                    /        |        \
                 WAIT      REPLY    INITIATE
                             |         |
                             +----+----+
                                  |
                                  v
                           Memory Retrieval
                                  |
                                  v
                         Persona-conditioned LLM
                                  |
                                  v
                               MESSAGE
```

An LLM is not allowed to invent activity merely to keep the interaction engaging. `WAIT` is expected to dominate most wall-clock time.

## Layers

### 1. Ingest

Input adapters parse exports such as KakaoTalk text into `ChatMessage` records:

- timestamp
- sender
- text
- source (`REAL` or `SIMULATION`)
- message type
- metadata

The original source file is never modified.

### 2. Observable persona

Persona is decomposed rather than represented by a single vague paragraph:

- language/style profile
- conversational behavior
- temporal behavior
- retrievable memories
- relationship-specific context

The first implementation only extracts language and temporal profiles.

### 3. Memory

REAL memories and SIMULATION memories are separate namespaces. A generated event can never silently become evidence about what happened in reality.

Future storage schema:

```text
memory_id
source: REAL | SIMULATION
timestamp
participants
content
embedding
structured_tags
```

### 4. Temporal engine

The engine reasons over wall-clock time. Closing the app does not freeze the simulated relationship. On resume, it processes the elapsed interval and determines whether any scheduled or plausible actions occurred.

### 5. Action policy

The action policy has a deliberately small surface:

```text
WAIT
REPLY
INITIATE
```

Later versions may add `CONTINUE_TOPIC`, `START_NEW_TOPIC`, `REACT`, and `END_CONVERSATION`, but those should remain behavior decisions rather than prose-generation decisions.

### 6. Message generation

Only after a non-WAIT action is chosen does the language model run. Retrieval supplies similar historical exchanges and relevant event memories. The model generates a message consistent with the chosen action, evidence, and style profile.

## Local-first boundary

Raw conversation exports, embeddings, generated memories, and local databases belong under ignored local paths. Repository code and tests must use synthetic fixtures only.
