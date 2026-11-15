# RetTune

A reproducible testbed for profiling component swaps, score calibration, and latency tradeoffs in foundational hybrid search.

I built RetTune as both an empirical investigation and a practical tool, motivated by problems I've run into while upgrading my own RAG system on the job. It focuses on the foundation layer of RAG, specifically the components where subtle IR issues (e.g. score distortion and vector anisotropy) very subtly break retrieval; usually what people deem the less exciting stuff in comparison to orchestrating the agent loops that then use the retrieval. 

Every benchmark and figure in this repo can be audited and rerun via the [ADD REFERENCE TO HOW TO REPRODUCE SECTION| How to Reproduce] guide. The codebase is also built to be modular, so you can drop in your own datasets/signal methods/fusion logic/encoders to stress-test your own search stack.