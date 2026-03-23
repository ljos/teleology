# Social anarchic, mad, and misbehaving agents collaboratively solving tasks
# I will address each of these topics through the presentation.
import asyncio
import contextlib
import functools
import inspect
import math
import operator as op
import os
import random
import re
import sqlite3
import textwrap
import threading

from abc import ABC
from collections.abc import AsyncIterator, Iterable
from compression import zstd
from contextlib import AbstractAsyncContextManager
from datetime import datetime, timedelta
from itertools import chain, islice, repeat, starmap, zip_longest
from pathlib import Path

# I use typing and types, but I don't generally use any type checker.
# I just like to annotated the code with what I expect to be at the
# bounderies of my code.
from types import TracebackType
from typing import Annotated, Awaitable, Callable, Literal, Self

import azure.identity.aio as identity
import openai

from faker import Faker  # not strictly needed, but makes it more fun...
from openai import APIConnectionError, InternalServerError, RateLimitError
from openai.types.responses import Response, ToolParam
from openai.types.responses import ResponseFunctionToolCall as ToolCall
from openai.types.responses import ResponseInputParam as Message
from openai.types.responses import ResponseInputText as InputText
from openai.types.responses import ResponseReasoningItem as ReasoningItem
from openai.types.responses.response_input_item import (
    FunctionCallOutput as ToolResponse,
)
from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from pydantic.json_schema import GenerateJsonSchema

# We could probably do without pydantic, but the openai library
# already needs it... an interesting note is that pythons standard
# library is LARGE and quite useful.


MODEL_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
MODEL_DEPLOYMENT = os.environ["AZURE_OPENAI_MODEL"]

DATABASE = os.environ.get("TELE_DATABASE", "data.sqlite")
DATA_DIR = Path(os.environ.get("TELE_DATA_DIR", "data")).absolute()

MAX_STEPS = os.environ.get("TELE_MAX_STEPS", 100)
MAX_VOTES = os.environ.get("TELE_MAX_VOTES", 3)
MAX_RETRY_LIMIT = os.environ.get("TELE_RETRY_LIMIT", 5)


# MAD agents Each agent divides its task into sub-tasks and requests
# "specialist" agents to solve the new sub-task, making the actual
# task each agent solves small.
# ref: https://arxiv.org/html/2511.09030v1

SYSTEM_PROMPT = """\
You are a helpful agent.

Any data you have access to is in the data folder in the current
working directory. The data folder is read only, but you can write to
the current working directory. All agents have access to the data.

The tasks you are given can be very complex, so follow these steps:

  - Investigate your notebook
  - Formulate a plan for how to solve the task
  - Note down interesting information in your notebook
  - Divide the task into sub-tasks
  - Note down acceptance criteria for your sub-tasks in your notebook
  - Request specialist agents do the sub-tasks
  - Compile a report from your findings

The user will not be available before you complete the task. If you
need help to interpret the task you should ask a specialist.

Try to delegate as many tasks as you can, unless it is a single simple
task that you can solve immediately.
 """

# I didn't want to implement user and agent feedback. It could be a
# cool thing to continue with....


class Attempt(AbstractAsyncContextManager):
    def __init__(self, lock: asyncio.Lock, retry: int) -> None:
        self.lock = lock
        self.retry = retry

        self.exception = None

    async def __aenter__(self) -> Self:
        # The lock is released after entering the context.
        async with self.lock:
            return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        self.exception = exc_val

        if exc_type is None:
            return

        if isinstance(exc_val, ValidationError):
            return True

        if isinstance(exc_val, APIConnectionError | InternalServerError):
            await asyncio.sleep(1.0 + self.retry**2)
            return True

        if isinstance(exc_val, RateLimitError):
            wait = exc_val.body["message"]
            wait = re.search(r"Please retry after (\d+) second", wait)
            wait = float(wait.group(1) if wait else 0)

            now = datetime.now()

            if not wait:  # wait until the next whole minute
                wait = now.replace(second=0, microsecond=0)
                wait = wait + timedelta(minutes=1)
                wait = (wait - now).total_seconds()

            end = now + timedelta(seconds=wait + 0.1)

            async with self.lock:
                wait = max(0.0, (end - datetime.now()).total_seconds())
                await asyncio.sleep(wait)
                return True


class LLM:
    def __init__(self):
        self.client = None
        self.lock = None

        self._stack = None

    async def __aenter__(self) -> Self:
        self.lock = asyncio.Lock()

        self._stack = contextlib.AsyncExitStack()

        self.client = await self._stack.enter_async_context(
            openai.AsyncOpenAI(
                base_url=f"{MODEL_ENDPOINT}/openai/v1",
                api_key=identity.get_bearer_token_provider(
                    await self._stack.enter_async_context(  # <- this is why I use the stack
                        identity.DefaultAzureCredential()
                    ),
                    "https://cognitiveservices.azure.com/.default",
                ),
            )
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        tb: TracebackType | None,
    ):
        self.lock, self.client = None, None
        await self._stack.aclose()

    async def ask(
        self,
        messages: list[Message],
        tools: list[ToolParam],
        text_format: type[BaseModel] | None = None,
    ) -> Response:
        kwargs = {}
        if text_format:
            # I am not handling $ref here. I am not sure if that is
            # neccessary or not anymore... the openai-python converts
            # it, but we are just using simple types here so we don't
            # have to worry about that...

            schema = text_format.model_json_schema()
            schema["additionalProperties"] = False
            required = schema["required"] = []
            for prop in schema["properties"].keys():
                required.append(prop)

            kwargs = {
                "text": {
                    "format": {
                        "name": text_format.__name__,
                        "type": "json_schema",
                        "strict": True,
                        "schema": schema,
                    }
                }
            }

        for i in range(1, MAX_RETRY_LIMIT + 1):
            async with Attempt(self.lock, i) as attempt:
                return await self.client.responses.create(
                    model=MODEL_DEPLOYMENT,
                    input=messages,
                    tools=tools,
                    tool_choice="auto",
                    parallel_tool_calls=True,
                    reasoning={
                        "effort": "low",
                        "summary": "concise",
                    },
                    include=["reasoning.encrypted_content"],
                    store=False,
                    **kwargs,
                )

        raise attempt.exception


class GenerateJsonSchema(GenerateJsonSchema):
    # Openai doesn't like the title field...
    def field_title_should_be_set(self, _) -> bool:
        return False


# It is cached because making typeadapters is really expensive. We
# could maybe make the schema ourselves, but this is just so easy when
# we already have pydantic right there!
@functools.cache
def schema(fn: Tool) -> ToolParam:
    return {
        "type": "function",
        "name": fn.__name__,
        "description": inspect.getdoc(fn),
        "strict": True,
        "parameters": TypeAdapter(fn).json_schema(
            mode="serialization",
            schema_generator=GenerateJsonSchema,
        ),
    }


class Database:
    def __init__(self, database: str):
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(database, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def create_function(self, *args, **kwargs):
        with self.lock:
            return self.conn.create_function(*args, **kwargs)

    def execute(self, query: str, params: tuple | list[tuple] = ()):
        with self.lock:
            value = self.conn.execute(query, params).fetchall()
            self.conn.commit()
            return value

    def executemany(self, query: str, params: tuple | list[tuple] = ()):
        with self.lock:
            value = self.conn.executemany(query, params).fetchall()
            self.conn.commit()
            return value


# Make a global variable and just lock everything! We want everything
# to run in the same process anyway. In a production setting we would
# do this differently.
db = Database(DATABASE)


# Just to generate names for the agents.
fake = Faker(["no_NO", "en_GB", "it_IT", "fr_FR", "de_DE"])


# An agent is nothing more than an id and a name. Everything else is
# just state that belongs somewhere else.
class Agent(BaseModel):
    id: int
    name: str


# Setup the database and find the first agent.
def init_app() -> Agent:
    db.execute("PRAGMA journal_mode=WAL")

    # We want to use sampling to find agents--easiest if we can do it
    # in the database.
    db.create_function("betavariate", 2, random.betavariate, deterministic=False)

    db.execute(
        """
        CREATE TABLE IF NOT EXISTS agent (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # Each agent has a belief about how well another agent will
    # perform based on what they themselves have observed of successes
    # and failures.
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS belief (
           id INTEGER PRIMARY KEY,
           agent_id INTEGER NOT NULL,
           target_id INTEGER NOT NULL,

           success INTEGER NOT NULL DEFAULT 0,
           failure INTEGER NOT NULL DEFAULT 0,

           FOREIGN KEY (agent_id) REFERENCES agent (id),
           FOREIGN KEY (target_id) REFERENCES agent (id)
        )
        """
    )
    db.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_belief_agent_target ON belief (agent_id, target_id)
        """
    )
    db.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_belief_agent_id ON belief (agent_id)
        """
    )
    db.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_belief_target_id ON belief (target_id)
        """
    )

    # Each agent keeps a notebook
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS note (
            id INTEGER PRIMARY KEY,
            agent_id INTEGER NOT NULL,

            title TEXT NOT NULL,
            content TEXT NOT NULL,

            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            visited_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (agent_id) REFERENCES agent(id)
        )
        """
    )

    # That can be searched
    db.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS note_fts USING fts5(
            title,
            content,
            content='note',
            content_rowid='id',
            tokenize='porter'
        )
        """
    )
    db.execute(
        """
        CREATE TRIGGER IF NOT EXISTS note_after_insert INSERT ON note BEGIN
            INSERT INTO note_fts(rowid, title, content) VALUES (new.id, new.title, new.content);
        END
        """
    )
    db.execute(
        """
        CREATE TRIGGER IF NOT EXISTS note_after_delete DELETE ON note BEGIN
            INSERT INTO note_fts(note_fts, rowid, title, content) VALUES ('delete', old.id, old.title, old.content);
        END
        """
    )
    db.execute(
        """
        CREATE TRIGGER IF NOT EXISTS note_after_update UPDATE ON note BEGIN
            INSERT INTO note_fts(note_fts, rowid, title, content) VALUES ('delete', old.id, old.title, old.content);
            INSERT INTO note_fts(rowid, title, content) VALUES (new.id, new.title, new.content);
        END
        """
    )

    # Make a couple of initial agents!
    n_agents = db.execute("SELECT count(*) FROM agent")[0][0]
    if n := max(0, 10 - n_agents):
        db.executemany(
            "INSERT INTO agent (name) VALUES (?)",
            [(fake.name(),) for _ in range(n)],
        )

    # We use Thompson sampling to find a suitable agent
    # https://en.wikipedia.org/wiki/Thompson_sampling
    ident, name, p = db.execute(
        """
        WITH outcome (id, name, success, failure) AS (
          SELECT
            agent.id,
            agent.name,
            coalesce(sum(belief.success), 0) + 1,
            coalesce(sum(belief.failure), 0) + 1
          FROM agent
          LEFT JOIN belief ON agent.id = belief.target_id
          GROUP BY agent.id
        )
        SELECT id, name, betavariate(success, failure) as p
        FROM outcome
        ORDER BY p DESC
        LIMIT 1
        """
    )[0]

    # If our candidate has a probabilty of less than 50%, we create a
    # new one.
    if p < 0.5:
        name = fake.name()
        (ident,) = db.execute(
            "INSERT INTO agent (name) VALUES (:name) RETURNING id",
            {"name": name},
        )[0]

    return Agent(id=ident, name=name)


# Agent reward is not through a capitalist monetary reward system, but
# relies on trust and experience. Each agent chooses using sampling
# from observed success and failure among the other agents.
#
# Eventually this _should_ create specialisation, community, and
# collaboration without hierarchy.
#
# A key component of other agent systems that I have seen either
# relies on some form of tokens and on a single leader/orchestrator
# that is responsible for keeping track of work. Here all agents are
# treated as equal and can be a orchestrator.


# A particular agent can also use Thompson sampling to find a suitable
# candidate to delegate a task to.
def sample(agent: Agent) -> Agent:
    ident, name, p = db.execute(
        """
        WITH outcome (id, name, success, failure) AS (
            SELECT
                agent.id,
                agent.name,
                coalesce(sum(belief.success), 0) + 1,
                coalesce(sum(belief.failure), 0) + 1
            FROM agent LEFT JOIN belief ON agent.id = belief.target_id
            WHERE agent.id = :id
            GROUP BY agent.id
        )
        SELECT id, name, betavariate(success, failure) AS p
        FROM outcome
        ORDER BY p DESC
        LIMIT 1
        """,
        {"id": agent.id},
    )[0]

    if p < 0.5:
        for _ in range(10):
            with contextlib.suppress(sqlite3.IntegrityError):
                name = fake.name()
                (ident,) = db.execute(
                    "INSERT INTO agent (name) VALUES (:name) RETURNING id",
                    {"name": name},
                )[0]
                break

    return Agent(id=ident, name=name)


# After delegation, the agent needs to update its belief.
def update_belief(
    agent: Agent,
    target: Agent,
    status: Literal["success", "failure"],
) -> None:
    db.execute(
        """
        INSERT OR IGNORE INTO belief (agent_id, target_id, success, failure)
        VALUES (:aid, :tid, 0, 0)
        """,
        {"aid": agent.id, "tid": target.id},
    )
    db.execute(
        """
        UPDATE belief
        SET success = success + :success,
            failure = failure + :failure
        WHERE agent_id = :aid AND target_id = :tid
        """,
        {
            "aid": agent.id,
            "tid": target.id,
            "success": int(status == "success"),
            "failure": int(status == "failure"),
        },
    )


def insert_note(agent: Agent, title: str, content: str) -> int:
    (ident,) = db.execute(
        """
        INSERT INTO note (agent_id, title, content)
        VALUES (:aid, :title, :content)
        RETURNING id
        """,
        {"aid": agent.id, "title": title, "content": content},
    )[0]
    return ident


def delete_note(agent: Agent, id: int) -> int:
    db.execute(
        """
        DELETE FROM note
        WHERE agent_id = :aid AND id = :nid
        RETURNING id
        """,
        {"aid": agent.id, "nid": id},
    )
    return id


def get_note(id: int) -> int:
    note = db.execute(
        """
        UPDATE note
        SET visited_at = CURRENT_TIMESTAMP
        WHERE id = :id
        RETURNING id, title, content
        """,
        {"id": id},
    )

    return note[0]


def get_latest_note_titles(agent: Agent, n=10):
    notes = db.execute(
        """
        SELECT id, title
        FROM note
        WHERE agent_id = :id
        ORDER BY visited_at DESC
        LIMIT :n
        """,
        {"id": agent.id, "n": n},
    )
    return notes


# Only BM25 search--this should probably be tuned to the kind of
# searches that the agents does and it probably wouldn't hurt to
# include nearest-neighbour vector search as well. I kept to bm25 as
# it is is built in to sqlite and I didn't need any more imports...
def search_notes(query: str) -> list[tuple[int, str, str]]:
    return db.execute(
        """
        SELECT
            note.id AS id,
            snippet(note_fts, 0, '<b>', '</b>', '...', 50) AS title,
            snippet(note_fts, 1, '<b>', '</b>', '...', 50) AS content
        FROM note
        JOIN note_fts AS s ON note.id = s.rowid
        WHERE note_fts MATCH ?
        ORDER BY bm25(note_fts, 10.0, 1.0) ASC
        LIMIT 10
        """,
        (query,),
    )


# These classes are strictly not neccessary, but since we have
# pydantic anyway, we can make the management of input and output from
# the model easier by defining and using them.
class NoteId(BaseModel):
    id: int


class Note(BaseModel):
    id: int
    title: str
    content: str


class Hit(BaseModel):
    note_id: int
    title: str
    content: str


class Hits(BaseModel):
    hits: list[Hit]


# This is pure convenience and so that we can see what is going on in
# the network.
class StdoutMixin(ABC):
    def stdout(self, content: str | Iterable[str], sep: str = "|"):
        content = [content] if isinstance(content, str) else content
        content = "\n".join(content)
        content = textwrap.indent(content, f"{self.agent.id:04d}{sep} ")
        print(content)


class Failed(BaseModel):
    error: str


# The notebook is necessary as that is the premise for the
# spesialisation of the agents. It is how the agent has an opportunity
# to affect its own prompt. It will do that by the summary in the
# notebook and through search and getting notes. You can also think
# about this as a way to introduce skills. This could be made more
# explicit by also adding topics to the notes.
class Notebook(StdoutMixin):
    def __init__(self, agent: Agent):
        self.agent = agent

    async def summary(self) -> AsyncIterator[str]:
        """
        A summary is in our case the id and title of the n latest notes.
        """
        notes = await asyncio.to_thread(get_latest_note_titles, self.agent, 10)

        summary = ["<notebook>", "NOTE_ID\tTITLE"]
        summary.extend(f"{id}\t{title}" for id, title in notes)
        summary.append("</notebook>\n")
        yield "\n".join(summary)

    async def add_note(
        self,
        title: Annotated[str, Field(description="Short description of the content")],
        content: Annotated[str, Field(description="The content of the note")],
    ) -> NoteId:
        """
        Add a note to your notebook.

        You should add anything that you need to remember the next
        time you are working on a similar task as a note. It might be
        a long time before you will work on the same type of task
        again so you will probably forget.

        The note should contain a title and the content of
        the note.

        The title should be a short description that tell something
        about what is in the note.

        The content should be as concise as possible. You should
        rather make two notes and use a reference to link them than
        write too much in the same note. You can link notes by adding
        Note#ID as a reference at the end of the content. This will make
        it easier to find notes later.

        Each reference should have a short sentence that describes it.
        """

        ident = await asyncio.to_thread(insert_note, self.agent, title, content)

        self.stdout([f"Note#{ident} {title}", content], sep="<")

        return NoteId(id=ident)

    async def remove_note(
        self,
        id: Annotated[int, Field(description="The id of the note to remove")],
    ) -> NoteId | None:
        """
        If it turns out a note is no longer useful or contains
        false information, you can try to delete it. If you are not
        the original owner of the note, the deletion will fail.

        Deleting notes should be considered very carefully, even if
        the content is not valuable now, it might be useful later.
        """

        id = await asyncio.to_thread(delete_note, self.agent, id)

        self.stdout(f"Note#{id if id else '~'}", sep="~")

        return NoteId(id=id)

    async def get_note(
        self,
        id: Annotated[int, Field(description="The id of the note to look up")],
    ) -> Note | None:
        """
        If you have the ID of a note, you can use this function to
        retrieve that note.
        """

        self.stdout(f"GET Note#{id}", sep=">")
        try:
            ident, title, content = await asyncio.to_thread(get_note, id)

            self.stdout([f"Note#{ident} {title}", content], sep=">")

            return Note(id=ident, title=title, content=content)
        except TypeError:
            pass

    async def search(
        self,
        query: Annotated[
            str, Field(description="The query to use to search for notes")
        ],
    ) -> Hits | Error:
        """
        You can query the notebook to find useful information that
        you or another agent has previously judged to be important for
        future tasks. You can use the match syntax in sqlite fts5.

        The search will return a list of hits. If any of the hits look
        interesting, you should get the full note using the id and the
        get_note function.

        The search uses the fts5 extension in sqlite.
        """
        query = query.replace("-", " ")
        self.stdout(f"SEARCH {query!r}", sep="$")

        try:
            hits = await asyncio.to_thread(search_notes, query)
        except sqlite3.OperationalError as e:
            return Failed(error=str(e))

        hits = [
            Hit(note_id=id, title=title, content=content)
            for (id, title, content) in hits
        ]

        for hit in hits:
            self.stdout([f"Note#{hit.note_id} {hit.title} {hit.content}"], sep="$>")

        return Hits(hits=hits)


class Code(BaseModel):
    code: str


class Result(BaseModel):
    output: str
    stdout: str
    stderr: str


class Error(BaseModel):
    error: str
    stdout: str
    stderr: str


OutputT = Result | Error
Output = TypeAdapter(OutputT)


# The execution of python is what makes the agent general and able to
# do "anything".
# Python runs in pyodide in deno. It just allows us to use a
# sandbox. The same could be done through containers or some other
# technology. This was just the easistest for me at the time of
# writing it as it required the least amount of setup and comes
# with some default packages that we can use/activate.
class Python(StdoutMixin):
    command: list[str] = [
        "deno",
        "run",
        "--no-prompt",
        "--cached-only",
        "--deny-net",
        f"--allow-read={Path('node_modules').absolute()},{DATA_DIR}",
        Path("src/sandbox/run.ts").absolute(),
    ]

    def __init__(self, agent: Agent):
        self.agent = agent
        self.process = None
        self.lock = asyncio.Lock()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        tb: TracebackType | None,
    ):
        if self.process:
            self.process.terminate()
            await self.process.wait()

    async def execute_python(
        self, code: Annotated[str, Field(description="The python code to execute")]
    ) -> OutputT:
        """
        Execute python code. Returns the output of the code execution.

        Available python packages:
          - matplotlib
          - networkx
          - numpy
          - pandas
          - scipy
          - uncertainties

        YOU CANNOT INSTALL ANY OTHER PACKAGES.
        THERE IS NO NETWORK ACCESS.
        """

        self.stdout(code, sep="!")

        data = Code(code=code)
        data = data.model_dump_json()
        data = data.encode()

        # We allow parallel tool calls, but we shouldn't allow parallel python calls...
        async with self.lock:
            # Small memory optimisation here... if the agent never
            # uses the process we should not create the expensive
            # process
            if not self.process:
                self.process = await asyncio.create_subprocess_exec(
                    *self.command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    limit=128 * 1024 * 1024,  # The output can be quite big...
                )

            self.process.stdin.writelines([data, b"\n"])
            await self.process.stdin.drain()

            output = await self.process.stdout.readline()

        output = output.decode()

        try:
            output = Output.validate_json(output)
        except ValidationError as e:
            return Failed(error=str(e))

        self.stdout(output.stdout, sep=")")
        self.stdout(output.stderr, sep="?")
        match output:
            case Result(output=value):
                self.stdout(value, sep=":")
            case Error(error=value):
                self.stdout(value, sep="?")

        return output


Frame = str | Callable[..., Awaitable[AsyncIterator[str]]]
Tool = Callable[..., Awaitable[BaseModel]]
Arguments = TypeAdapter(dict)


class ToolNotFoundError(Exception):
    pass


# The execution of an agent is not that complex actually! Lets think
# about it: What does an agent need to be functional and fully
# configurable?
class Runner(StdoutMixin):
    def __init__(
        self,
        llm: LLM,
        agent: Agent,
        frames: list[Frame],
        tools: list[Tool],
    ):
        self.llm = llm
        self.agent = agent  # just for pretty printing
        self.frames = frames
        self.tools = {tool.__name__: tool for tool in tools}

    @property
    def schema(self) -> list[ToolParam]:
        return [schema(fn) for fn in self.tools.values()]

    async def context(self) -> AsyncIterator[InputText]:
        for frame in self.frames:
            if isinstance(frame, str):
                yield InputText(text=frame, type="input_text")
                continue

            async for value in frame():
                yield InputText(text=value, type="input_text")

    async def apply(self, fc: ToolCall) -> ToolResponse:
        try:
            fn = self.tools[fc.name]
            arguments = Arguments.validate_json(fc.arguments)
            output = await fn(**arguments)

            return ToolResponse(
                call_id=fc.call_id,
                output=output.model_dump_json(),
                type="function_call_output",
            )
        except KeyError as e:
            raise ToolNotFoundError(f"Tool not found {fc.name!r}") from e

    async def run(
        self,
        user: Agent,  # <- also for pretty printing
        task: str,
        text_format: type[BaseModel] | None = None,
    ) -> BaseModel | str:
        self.stdout("system")
        self.stdout("---------")

        context = [frame async for frame in self.context()]

        for ctx in context:
            self.stdout(ctx.text)

        self.stdout(user.name)
        self.stdout("--------")
        print(f"{self.agent.id:03d}: {task}")

        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": task},
        ]

        for _ in range(MAX_STEPS):
            response = await self.llm.ask(messages, self.schema, text_format)
            messages.extend(response.output)

            for message in response.output:
                match message:
                    case ReasoningItem(summary=summary):
                        summary = "\n".join(s.text for s in summary)
                        self.stdout(summary, sep="*")

            results = response.output
            results = (fc for fc in results if fc.type == "function_call")
            results = (self.apply(fc) for fc in results)
            results = await asyncio.gather(*results)

            if not results:
                break

            messages.extend(results)

        return response


# No agent directly collaborates in a task, but they collaborate
# through voting and evaluation. Over time, the evaluation and
# updating of their belief of other agents success/failure will create
# communities of practice.
#
# This could be improved by allowing an agent to refuse a task.

# import tiktoken
# import struct
# encoding = tiktoken.get('o200k_base')


def vote(agents: Agents, solutions: list[str]) -> tuple[Agent, str]:
    # The solution with the lowest Normalized Compression Distance to
    # the other solutions wins the vote!
    #
    # ref: https://en.wikipedia.org/wiki/Normalized_compression_distance
    # ref: https://maxhalford.github.io/blog/text-classification-zstd/
    #
    # Alternative with embeddings: https://arxiv.org/abs/2402.05120

    idx, best = None, math.inf

    # data = map(str.casefold, solutions) ??
    data = [s.encode() for s in solutions]

    # Whould be interesting to test with and without token encoding and
    # look at what performs better. Could we encode phrases instead?

    # data = map(encoding.encode, solutions)
    # data = [struct.pack(f">{len(s)}I", *s) for s in data]

    for i, x in enumerate(data):
        c = zstd.ZstdCompressor()

        y_z = chain(data[:i], data[i + 1 :])
        y_z = zip_longest([], y_z, fillvalue=b"\n")
        y_z = islice(chain.from_iterable(y_z), 1)
        y_z = sum(len(b) for b in map(c.compress, y_z))
        y_z = y_z + len(c.flush(mode=c.FLUSH_BLOCK))

        x_z = len(zstd.compress(x))

        xy_z = y_z + len(c.compress(x)) + len(c.flush())

        dist = (xy_z - min(x_z, y_z)) / max(x_z, y_z)

        idx, best = min((idx, best), (i, dist), key=op.itemgetter(1))

    return agents[idx], solutions[idx]


EVAL_PROMPT = """\
Please evaluate if the tasks was solved in a satisfactory way.

Check your notebook for any acceptance criteria for the task.

TASK:
{task}

SOLUTION:
{solution}
"""


class Evaluation(BaseModel):
    status: Literal["failure", "success"]
    reason: str


class Solution(BaseModel):
    solution: str


# It can take a lot of memory and put a lot of contention on the llm
# if we run too many agents at the same time.
agent_limit = asyncio.Semaphore(value=50)


# This is a naive implementation based on promise theory. An agent
# promises that it will evaluate and accept an answer if the other
# agent promises to provide an answer.
#
# ref: https://markburgess.org/promises.html
class Orchestrator(StdoutMixin):
    def __init__(self, agent: Agent, llm: LLM):
        self.agent = agent
        self.llm = llm

    async def assign(self, agent: Agent, task: str) -> str:
        # I don't like the py context manger here
        async with agent_limit, Python(agent) as py:
            notebook = Notebook(agent)
            orchestrator = Orchestrator(agent, self.llm)

            runner = Runner(
                llm=self.llm,
                agent=agent,
                frames=[
                    SYSTEM_PROMPT,
                    notebook.summary,
                ],
                tools=[
                    notebook.add_note,
                    notebook.remove_note,
                    notebook.get_note,
                    notebook.search,
                    py.execute_python,
                    orchestrator.request,
                ],
            )

            response = await runner.run(self.agent, task)

            output = filter(None, (o.content for o in response.output))
            output = (v for o in output for v in o)
            output = filter(None, (getattr(t, "text", None) for t in output))
            output = "\n".join(output)

            return output

    async def evaluate(self, agent: Agent, task: str, solution: str) -> str:
        notebook = Notebook(self.agent)
        runner = Runner(
            llm=self.llm,
            agent=self.agent,
            frames=[
                SYSTEM_PROMPT,
                notebook.summary,
            ],
            tools=[
                notebook.get_note,
                notebook.search,
            ],
        )

        evaluation = await runner.run(
            agent,
            EVAL_PROMPT.format(task=task, solution=solution),
            text_format=Evaluation,
        )

        self.stdout(f"{agent.name} - {evaluation.status}", sep="%")
        await asyncio.to_thread(update_belief, self.agent, agent, evaluation.status)

        try:
            evaluation = evaluation.output[1].content[0].text
            evaluation = Evaluation.model_validate_json(evaluation)
            return evaluation.reason
        except ValidationError:
            return "failure"

    # This is also a tool! It can also be thought of a as a step in
    # behavioural programming, Each b-thread agent cannot directly
    # block other behaviours, but the there are requests of work and
    # some events are blocked by not being voted as next.
    # ref: https://lmatteis.github.io/react-behavioral/
    async def request(
        self,
        task: Annotated[str, Field(description="Detailed description of the task")],
    ) -> Solution:
        """
        Request that a task is done by a specialist.
        """
        agents = repeat((sample, self.agent), MAX_VOTES)
        agents = starmap(asyncio.to_thread, agents)
        agents = await asyncio.gather(*agents)

        self.stdout(f"REQ {', '.join(a.name for a in agents)}", sep="#")
        self.stdout(task, sep="#")

        solutions = zip_longest(agents, [], fillvalue=task)
        solutions = starmap(self.assign, solutions)
        solutions = await asyncio.gather(*solutions)

        agent, solution = await asyncio.to_thread(vote, agents, solutions)

        await self.evaluate(agent, task, solution)

        return Solution(solution=solution)


async def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("TASK", type=str)

    args = parser.parse_args()

    async with LLM() as llm:
        orchestrator = Orchestrator(Agent(id=0, name="user"), llm)

        agent = await asyncio.to_thread(init_app)
        solution = await orchestrator.assign(agent, args.TASK)

        print(f"0000| {agent.name}")
        print("0000|------")
        text = textwrap.indent(solution, "0000| ")
        print(text)


if __name__ == "__main__":
    asyncio.run(main())
