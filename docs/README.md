# Documentation

Docs for **cluster_compilot** — a faithful implementation of ComPilot (LLM-guided loop optimization, [arXiv:2511.00592](https://arxiv.org/abs/2511.00592)).

| Doc | What's in it |
|---|---|
| [Architecture](architecture.md) | The four diagrams (rendered images + live mermaid): system, dialogue, legality, backends |
| [How it works](how-it-works.md) | The evaluation pipeline (parse → θ′ → ISL legality → measure), feedback categories, the 9-primitive DSL |
| [Building (step by step)](building.md) | Prereqs → deps → key → smoke test → optional Tiramisu build |
| [User guide (step by step)](user-guide.md) | Run the agent, eval, benchmark, tests; write schedules; add a kernel |
| [Parallelism (Python 3.14)](parallelism.md) | Thread fan-out (best-of-k, candidates/turn), the two locks, parallelize-search-serialize-measurement |
| [Test results](test-results.md) | Real test + benchmark output, with screenshots |

![architecture](images/architecture.png)
