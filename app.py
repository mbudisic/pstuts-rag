import asyncio
import logging
import random
import signal
import time
from datetime import datetime
from typing import cast
from uuid import uuid4

import chainlit as cl
import httpx
import nest_asyncio
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from chainlit.input_widget import Select

from pstuts_rag.configuration import Configuration
from pstuts_rag.datastore import Datastore
from pstuts_rag.nodes import (
    FinalAnswer,
    TutorialState,
    initialize,
    YesNoDecision,
)
from pstuts_rag.utils import get_unique
from version import __version__

# Track the single active session
active_session = {"id": None, "timestamp": None}

load_dotenv()
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.INFO)

# Apply nested asyncio to enable nested event loops
nest_asyncio.apply()

# Generate a unique ID for this application instance
unique_id = uuid4().hex[0:8]

# TODO: Create an introduction message here that explains the purpose of the app


async def sample_prompt_send(action: cl.Action):
    # Simulate a user message using the payload
    msg = cl.Message(content=action.payload["text"], author="user")
    await msg.send()
    time.sleep(0.5)
    await message_handler(msg)  # send the message to LLM for response


sample_prompts = {
    "sample_layers": "What are layers?",
    "sample_lasso": "How do I use lasso when the background is very busy?",
    "sample_paris": "What is the capital of France?",
}

for sample_label in sample_prompts:
    sample_prompt_send = cl.action_callback(sample_label)(sample_prompt_send)


@cl.on_chat_start
async def on_chat_start():
    """
    Handler for the start of a new chat session in Chainlit.

    - Initializes the application state for a new user session.
    - Sets up configuration, unique thread/session IDs, and the vector database (datastore).
    - Triggers asynchronous population of the datastore from transcript files.
    - Compiles the AI graph and stores all session objects in Chainlit's user session context.
    - Notifies the user that the session is active.
    """

    global active_session
    session_id = cl.context.session.id
    current_time = datetime.now()

    sample_actions = [
        cl.Action(
            name=name,
            label='"%s"' % text,
            payload={"text": text},
            icon="mouse-pointer-click",
        )
        for name, text in sample_prompts.items()
    ]

    await cl.Message(
        content=f"🟢 Session ID: `{session_id[:8]}` active.",
        author="System",
    ).send()

    # Eva introduction message
    await cl.Message(
        content=(
            f"👋 Hi there! I'm **Eva v.{__version__}**, your friendly Photoshop expert AI.\n\n"
            "---\n\n"
            "I was created as the demo app for [AI Makerspace](aimakerspace.io) Cohort 6 using Adobe Research [PsTuts data](https://github.com/adobe-research/PsTuts-VQA-Dataset).\n"
            "I'm here to help you with all your Photoshop questions, using real answers from training video transcripts. 🎥✨\n\n"
            "**How I work:**\n"
            "- I answer using only what's in the official training videos and in Adobe Help website.\n"
            "- You can decide to let me use the Adobe Help or not (or ask every time) using the gear button next to the prompt textbox.\n"
            "- If I find the answer, I'll include the timestamp so you can jump right to it! ⏱️\n"
            "- If it's not covered, I'll let you know honestly—no guessing, no made-up info.\n\n"
            "Feel free to ask anything about Photoshop, and let's get creative together! 🖼️🖱️\n"
            "Click on the following buttons to try out some sample prompts:\n"
        ),
        actions=sample_actions,
        author="Eva",
    ).send()

    # Deactivate any previous session
    active_session = {"id": session_id, "timestamp": current_time}

    configuration = Configuration()
    thread_id = f"chat_{uuid4().hex[:8]}"
    configuration.thread_id = thread_id

    cl.user_session.set(
        "eva_search_permission", configuration.search_permission
    )

    # Map permission to index for initial selection
    permission_to_index = {"ask": 0, "yes": 1, "no": 2}
    initial_index = permission_to_index.get(configuration.search_permission, 2)

    await cl.ChatSettings(
        [
            Select(
                id="eva_search_permission",
                label="Web Search Permission",
                values=["ask", "yes", "no"],
                initial_index=initial_index,
            )
        ]
    ).send()

    # Instantiate the Datastore and register a callback to notify when loading is complete
    datastore = Datastore(config=configuration)
    datastore.add_completion_callback(
        lambda: cl.run_sync(
            cl.Message(content="Datastore loading completed.").send()
        )
    )

    checkpointer = MemorySaver()
    # Compile the AI graph synchronously (blocking as intended)
    datastore, graph_builder = initialize(datastore)
    ai_graph = graph_builder.compile(checkpointer=checkpointer)

    # Start async population of the datastore from transcript files
    asyncio.create_task(
        datastore.from_json_globs(configuration.transcript_glob)
    )

    cl.user_session.set("configuration", configuration)
    cl.user_session.set("datastore", datastore)
    cl.user_session.set("checkpointer", checkpointer)
    cl.user_session.set("ai_graph", ai_graph)
    cl.user_session.set("thread_id", thread_id)


def format_video_reference(doc: Document):
    """
    Format a video reference from a LangChain Document into a Chainlit message with a video element.

    Args:
        doc (Document): The document containing video metadata (title, source, start, stop).

    Returns:
        cl.Message: Chainlit message with a video preview and timestamp.
    """
    v = {k: doc.metadata[k] for k in ("title", "source", "start", "stop")}

    v["start_min"] = f"{round(v['start'] // 60)}m:{round(v['start'] % 60)}s"
    video_link = cl.Video(
        name=v["title"],
        url=f"{v['source']}#t={v['start']}",
        display="side",
    )
    video_message = cl.Message(
        content=f"📼 Watch {video_link.name} (_@ {v['start_min']}_)",  # text has to include video name
        elements=[video_link],
    )

    return video_message


async def format_url_reference(url_ref):
    """
    Asynchronously fetch a screenshot preview for a URL using the Microlink API and format it as a Chainlit message.

    Args:
        url_ref (dict): Dictionary with 'url' and 'title' keys.

    Returns:
        cl.Message: Chainlit message with a screenshot image (if available) and a clickable link.
    """
    microlink = "https://api.microlink.io"
    params = {
        "url": url_ref["url"],
        "screenshot": {
            "overlay": {
                "background": "linear-gradient(225deg, #FF057C 0%, #8D0B93 50%, #321575 100%)",
                "browser": "dark",
            }
        },
    }

    screenshot = None
    async with httpx.AsyncClient() as client:
        try:
            payload = await client.get(microlink, params=params, timeout=30.0)
            if payload:
                logging.info(f"Successful screenshot\n{payload.json()}")
                screenshot = cl.Image(
                    name=f"{url_ref['title']}",
                    display="side",  # Show in the sidebar
                    url=payload.json()["data"]["screenshot"]["url"],
                    content=f"🔗 {url_ref['title']} [(go to website))]({url_ref['url']})",
                )
        except Exception as e:
            logging.error(f"Error fetching screenshot: {e}")

    return cl.Message(
        content=f"🔗 {url_ref['title']} [(go to website)]({url_ref['url']})",
        elements=([screenshot] if screenshot else []),
    )


class ChainlitCallbackHandler(BaseCallbackHandler):
    """
    Custom callback handler for Chainlit to visualize the execution of LangChain chains/graphs.

    - Tracks the current step in the graph and displays it in the Chainlit UI.
    - Handles step start, end, and error events, ensuring the UI is updated accordingly.
    """

    def __init__(self):
        self.current_step = None
        self.step_counter = 0

    # TODO: Make the step label update instead of add
    async def on_chain_start(self, serialized, inputs, **kwargs):
        """
        Called when a new chain/graph step starts.
        Creates a new Chainlit step for visualization if the step is part of the graph.
        """
        try:
            logging.info(kwargs)
            if (
                "name" in kwargs
                and "tags" in kwargs
                and len(list(filter(lambda t: "graph" in t, kwargs["tags"])))
                > 0
            ):
                self.step_counter += 1
                node_name = kwargs["name"]
                self.current_step = cl.Step(
                    name=f"{node_name} (step {self.step_counter})"
                )
                await self.current_step.__aenter__()

        except Exception as e:
            # If step creation fails, still increment counter and create a fallback step
            self.step_counter += 1
            print(f"Error in on_chain_start: {e}")
            self.current_step = cl.Step(
                name=f"Exception step_{self.step_counter}"
            )
            await self.current_step.__aenter__()

    async def on_chain_end(self, outputs, **kwargs):
        """
        Called when a chain/graph step ends.
        Closes the Chainlit step and optionally attaches output.
        """
        try:
            if self.current_step:
                # Optional: Add output to the step
                if outputs:
                    self.current_step.output = str(outputs)

                # Close the step - this stops the flashing/loading state
                await self.current_step.__aexit__(None, None, None)
                self.current_step = None

        except Exception as e:
            print(f"Error in on_chain_end: {e}")
            # Even if there's an error, try to close the step
            if self.current_step:
                try:
                    await self.current_step.__aexit__(None, None, None)
                    self.current_step = None
                except:
                    # Suppress all exceptions here to avoid cascading errors
                    pass

    async def on_chain_error(self, error, **kwargs):
        """
        Called when a chain/graph step raises an error.
        Closes the Chainlit step and attaches the error message.
        """
        try:
            if self.current_step:
                self.current_step.output = f"Error: {str(error)}"
                await self.current_step.__aexit__(None, None, None)
                self.current_step = None
        except Exception as e:
            print(f"Error in on_chain_error: {e}")


async def handle_interrupt(query: str) -> YesNoDecision:

    try:
        user_input = await cl.AskActionMessage(
            content="Search has been interrupted. Do you approve query: '%s' to be sent to Adobe Help?"
            % query,
            timeout=30,
            raise_on_timeout=True,
            actions=[
                cl.Action(
                    name="approve",
                    payload={"value": "yes"},
                    label="✅ Approve",
                ),
                cl.Action(
                    name="cancel",
                    payload={"value": "cancel"},
                    label="❌ Cancel web search",
                ),
            ],
        ).send()
        if user_input and user_input.get("payload").get("value") == "yes":
            return YesNoDecision(decision="yes")
        else:
            return YesNoDecision(decision="no")

    except TimeoutError:
        await cl.Message(
            "Timeout: No response from user. Canceling search."
        ).send()
        return YesNoDecision(decision="no")


from pstuts_rag.nodes import YesNoDecision


@cl.on_message
async def message_handler(input_message: cl.Message):
    """
    Main message handler for incoming user messages in Chainlit.

    - Checks if the session is active; if not, notifies the user and aborts.
    - Retrieves the AI graph and configuration from the session context.
    - Invokes the AI graph asynchronously with the user's query.
    - Streams the final answer token-by-token to the user.
    - Sends any video or URL references as additional messages.

    Args:
        input_message (cl.Message): The incoming user message from the chat UI.
    """
    global active_session
    current_session_id = cl.context.session.id

    # Check if this is the active session; only one session is allowed at a time
    if current_session_id != active_session["id"]:
        await cl.Message(
            content="🔴 **Inactive Session**\n\nThis tab is no longer active. Please close this tab and use the active session.",
            author="System",
        ).send()
        return  # Don't process the message

    ai_graph = cast(Runnable, cl.user_session.get("ai_graph"))
    configuration = cl.user_session.get("configuration")

    if not configuration:
        await cl.Message(content="Error: Configuration not found").send()
        return

    # Convert Configuration to RunnableConfig format and attach callback handler for Chainlit visualization
    config = configuration.to_runnable_config()
    config["callbacks"] = [ChainlitCallbackHandler()]

    raw_response = await ai_graph.ainvoke(
        {"query": input_message.content}, config
    )

    if "__interrupt__" in raw_response:
        logging.warning("*** INTERRUPT ***")

        logging.info(raw_response["__interrupt__"])

        answer: YesNoDecision = await handle_interrupt(
            raw_response["__interrupt__"][-1].value["query"]
        )

        raw_response = await ai_graph.ainvoke(
            Command(resume=answer.decision), config
        )

    response = cast(TutorialState, raw_response)

    # Start formatting tasks early to maximize concurrency.
    # Video reference formatting is synchronous, so we just collect the messages.
    # URL reference formatting is asynchronous (may involve network I/O), so we schedule those as tasks.
    # By starting the async tasks before streaming the answer, we allow them to run in the background while the answer is being streamed,
    # reducing the total perceived latency for the user.
    video_reference_messages = [
        format_video_reference(v)
        for v in get_unique(response["video_references"])
    ]
    url_reference_tasks = [
        asyncio.create_task(format_url_reference(u))
        for u in get_unique(response["url_references"])
    ]

    # Stream the final answer token-by-token for a typing effect
    for msg in response["messages"]:
        if isinstance(msg, FinalAnswer):
            final_msg = cl.Message(content="", author=msg.type)
            await final_msg.send()
            tokens = list(msg.content)
            for token in tokens:
                await final_msg.stream_token(token)
                time.sleep(0.02 / random.uniform(1, 10))
            if final_msg:
                await final_msg.update()

    # After streaming the answer, display video references (synchronous)
    await cl.Message(
        content=f"Formatting {len(response['video_references'])} video references."
    ).send()
    for msg in video_reference_messages:
        await msg.send()

    # Await and display URL references (asynchronous)
    await cl.Message(
        content=f"Formatting {len(response['url_references'])} website references."
    ).send()
    url_reference_messages = await asyncio.gather(*url_reference_tasks)
    for msg in url_reference_messages:
        await msg.send()


@cl.on_chat_end
async def end():
    """
    Handler for the end of a chat session in Chainlit.
    Logs the session end event.
    """
    session_id = cl.context.session.id
    logging.info(f"Session ended: {session_id}")


# 2. Update Configuration on settings update
@cl.on_settings_update
async def on_settings_update(settings):
    configuration = cl.user_session.get("configuration")
    if configuration and "eva_search_permission" in settings:
        configuration.search_permission = settings["eva_search_permission"]
        cl.user_session.set("configuration", configuration)
        cl.user_session.set(
            "eva_search_permission", settings["eva_search_permission"]
        )


if __name__ == "__main__":

    def handle_sigint(signum, frame):
        """
        Handle SIGINT (Ctrl+C) gracefully by printing a message and exiting.
        """
        print("SIGINT received (Ctrl+C), exiting...")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)
    pass
