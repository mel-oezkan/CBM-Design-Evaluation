# cbm-eval

--8<-- "README.md:intro"

Use it to ask questions such as: does LLM concept discovery help worst-group accuracy compared to
annotated concepts? How much do sequential and joint training differ in concept leakage? Which
filtering step actually matters? Each question becomes a sweep over one or more stages, and the
analysis reports the effect of every choice with confidence intervals.

!!! tip "New here?"
    Start with [Getting started](getting-started.md). It installs the package, trains a CBM and
    runs a small study on the offline synthetic dataset in a few minutes.

<div class="grid cards" markdown>

-   **Architecture**

    ---

    How a config becomes a trained CBM: the [pipeline map](architecture.md), the objects stages
    exchange, and the [component catalogue](components.md) of every registered variant and its kwargs.

-   **Experiments**

    ---

    Turning a design question into a sweep: [anchors, ablations and `run_id`](configs.md), instance
    bags, part localization, and the [Koh et al. 2020 reproduction](koh2020_reproduction.md).

-   **Extending**

    ---

    [Adding a component](contributing.md) without breaking the architecture: one registered
    subclass per variant, config kwargs as constructor kwargs, and an offline test.

</div>

--8<-- "README.md:status"
