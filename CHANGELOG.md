# Changelog

All notable changes to this project will be documented in this file.

## [1.3.0] - 2026-05-08

### Added

- Social dialogue layer with speech-act based interactions
- Meme market topic competition with heat/novelty/controversy dynamics
- Text-to-structured social signal extraction and feedback into agent state
- Social memory compression from episodic to procedural memory
- Unified world cup runner module for shared bootstrap logic
- Open-source repository scaffolding (`LICENSE`, contribution and issue templates, CI)

### Changed

- `memory_stream` moved to compatibility view on layered memory store
- LLM JSON handling made tolerant to fenced and embedded JSON
- Tactical non-interactive quick path avoids unnecessary LLM initialization

### Fixed

- Style archetype initialization order in agent bootstrap
- Decision memory step alignment with memory clock
- Forced write path for critical memory events
