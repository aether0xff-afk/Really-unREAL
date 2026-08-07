# Persona Model

## What persona means here

A persona is an evidence-backed model of **observable behavior**, not a claim about a person's private mental state.

We explicitly avoid latent labels such as:

```text
affection = 0.84
jealousy = 0.31
```

Those numbers look precise without being verifiable. Instead, we keep measurable features.

## Language profile

Initial features:

- message count
- mean and median character length
- short-message ratio
- multiline-message ratio
- `ㅋ`/laugh expression rate
- `ㅠ`/`ㅜ` expression rate
- frequent lexical tokens

Later features:

- message splitting into rapid consecutive bubbles
- punctuation habits
- emoji/sticker usage
- honorific/register changes
- topic-transition patterns
- question-return frequency

## Conversation behavior

Planned observable features include:

- who starts a new conversation session
- whether a question receives a question in return
- how often a topic is continued vs changed
- common conversation-ending forms
- number of consecutive messages per turn

## Relationship-specific conditioning

The target is not `P(next message | person)` in isolation. It is closer to:

```text
P(next observable action |
  this person's history,
  this conversation partner,
  recent dialogue,
  elapsed time,
  current context)
```

This matters because people can speak differently to different contacts.

## Evidence hierarchy

When generating behavior, prefer evidence in this order:

1. Similar exchanges between the same two participants.
2. Stable behavior measured across their shared history.
3. Explicit event/context memory.
4. Generic model prior only when data is sparse.

The system should expose uncertainty when evidence is sparse rather than inventing confidence.
