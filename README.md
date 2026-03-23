# TELEOLOGY

    Social anarchic, mad, and misbehaving agents collaboratively solving tasks


An experiment in multi-agentic systems that was presented at the 2026
Equinor Developer Conference. The idea is that the system should just
"do what I mean."


## Anarchic

I wanted didn't want the system to have a set leader or boss and I
didn't want to a pseudo monetary system. I therefore implemented trust
as a sampling strategy over the belief of an agent based on what the
agent has observed as sucess or failure.

The current implementation doesn't allow an agent to refuse a task.

## MAD

MAD or [Maximal Agentic
Decomposition](https://arxiv.org/abs/2511.09030) is a technique to
decompose the tasks into smaller tasks that can be more easily solved.
An orchestrator decomposes a tasks and gives the task to sub agents
that then vote which solution solves the tasks the best.

## Misbehaving

Behavioural programming is a programming paradigm that tries to make
it easier to "deal with underspecification and conflicting requirements."

I take some of the ideas from here, but it turned out difficult to
make it the engine for agentic behaviour.

I still believe there is some intersting stuff that can be done here,
but in this experiment it is only the request that is alive. One can
argue that the voting could be considered block... but it the whole
system doesn't continue to block..


- https://www.wisdom.weizmann.ac.il/~amarron/BP%20-%20CACM%20-%20Author%20version.pdf
- https://lmatteis.github.io/react-behavioral/



## Agents

The idea of trust and evaluation came partly from promise theory: https://markburgess.org/promises.html

Evaluation is local to each agent and trust is measured as success and
failure. Each agent promise to solve their assigned task.


## Tasks

The tasks for the system to solve that was presented was

    You will find a corpus detailing old norwegian books in the data
    folder. Please investigate the corpus and suggest and complete a
    number of different corpus linguistics analysis of the corpus.
    Look at how word distribution in poems change over time and indicates
    a change in topic. The report should be 10 pages. At the end of the
    report, present future work as a list of steps and instructions.

## How to run

First you need to ensure the sandbox is ready

```
deno install --frozen
deno run --allow-read --allow-net --allow-write src/sandbox/run.ts
```

This will install the deno dependencies and python dependencies for
pyodide. The python script only allows to read the cache and the data
folder and will fail if it tries to use any uninstalled packages.



The agent system can be then be invoked invoked by
```
uv run python -m teleology.mechanism "${TASK}"
```
