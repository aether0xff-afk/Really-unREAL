# Temporal Model

Time is a core part of the persona, not presentation metadata.

## Two different timing problems

### Reply timing

When another participant has already sent a message, estimate a reply-delay distribution conditioned on observable context.

Useful evidence includes:

- hour of day
- weekday/weekend
- whether a conversation is already active
- recent reply delays
- message type/length
- historical behavior in similar contexts

A distribution is preferable to a single mean. A person who sometimes replies in 20 seconds and sometimes in 40 minutes should not become a bot that always replies after 20 minutes.

### Initiation timing

Initiation is an event-occurrence problem, not just another reply delay.

Relevant evidence includes:

- baseline initiation frequency
- elapsed time since last contact
- time of day
- recent initiator balance
- unresolved topics
- remembered upcoming events
- prior session-start patterns

Most checks should produce `WAIT`.

## Initial measurable profile

Phase 1 extracts:

- active-hour histogram
- reply-delay samples when the target speaker follows another speaker
- median and interquartile reply delay
- conversation sessions separated by a configurable inactivity gap
- number/rate of sessions initiated by the target speaker

These are descriptive statistics. They are not yet a generative temporal model.

## Real-time semantics

Later live mode stores `last_processed_at`. If the app closes at time A and reopens at time B, the engine advances across the interval A→B. It must not pretend that only a few seconds passed.

Any spontaneous message generated during downtime receives its simulated event timestamp within that interval and is visible when the application resumes.
