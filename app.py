from pstuts_rag.configuration import Configuration
import asyncio
from typing import cast

import chainlit as cl
from dotenv import load_dotenv

from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable

from pstuts_rag.datastore import DatastoreManager
from pstuts_rag.nodes import initialize

import nest_asyncio
from uuid import uuid4

import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.INFO)

# Apply nested asyncio to enable nested event loops
nest_asyncio.apply()

# Generate a unique ID for this application instance
unique_id = uuid4().hex[0:8]


@cl.on_chat_start
async def on_chat_start():
    """
    Initializes the application when a new chat session starts.

    Sets up the language model, vector database components, and spawns tasks
    for database population and graph building.
    """

    configuration = Configuration()
    # Generate a unique thread_id for this chat session
    thread_id = f"chat_{uuid4().hex[:8]}"
    configuration.thread_id = thread_id

    datastore = await asyncio.to_thread(
        lambda: DatastoreManager(config=configuration).add_completion_callback(
            lambda: cl.run_sync(
                cl.Message(content="Datastore loading completed.").send()
            )
        )
    )

    checkpointer = MemorySaver()
    # Initialize and compile graph synchronously (blocking as intended)
    datastore, graph_builder = initialize(datastore)
    ai_graph = graph_builder.compile(checkpointer=checkpointer)

    asyncio.create_task(
        datastore.from_json_globs(configuration.transcript_glob)
    )

    cl.user_session.set("configuration", configuration)
    cl.user_session.set("datastore", datastore)
    cl.user_session.set("checkpointer", checkpointer)
    cl.user_session.set("ai_graph", ai_graph)
    cl.user_session.set("thread_id", thread_id)


# def process_response(
#     response_message: BaseMessage,
# ) -> Tuple[str, List[cl.Message]]:
#     """
#     Processes a response from the AI agents.

#     Extracts the main text and video references from the response,
#     and creates message elements for displaying video content.

#     Args:
#         response: Response object from the AI agent

#     Returns:
#         Tuple containing the text response and a list of message elements with video references
#     """
#     streamed_text = f"[_from: {response_message.name}_]\n"
#     msg_references = []

#     if response_message.name == VIDEOARCHIVE:
#         text, references = pstuts_rag.rag.RAGChainFactory.unpack_references(
#             str(response_message.content)
#         )
#         streamed_text += text

#         if len(references) > 0:
#             references = json.loads(references)
#             print(references)

#             for ref in references:
#                 msg_references.append(
#                     cl.Message(
#                         content=(
#                             f"Watch {ref['title']} from timestamp "
#                             f"{round(ref['start'] // 60)}m:{round(ref['start'] % 60)}s"
#                         ),
#                         elements=[
#                             cl.Video(
#                                 name=ref["title"],
#                                 url=f"{ref['source']}#t={ref['start']}",
#                                 display="side",
#                             )
#                         ],
#                     )
#                 )
#     else:
#         streamed_text += str(response_message.content)

#         # Find all URLs in the content
#         urls = re.findall(
#             r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\.-]*(?:\?[/\w\.-=&%]*)?",
#             str(response_message.content),
#         )
#         print(urls)
#         links = []
#         # Create a list of unique URLs
#         for idx, u in enumerate(list(set(urls))):

#             url = "https://api.microlink.io"
#             params = {
#                 "url": u,
#                 "screenshot": True,
#             }

#             payload = requests.get(url, params)

#             if payload:
#                 print(f"Successful screenshot\n{payload.json()}")
#                 links.append(
#                     cl.Image(
#                         name=f"Website {idx} Preview: {u}",
#                         display="side",  # Show in the sidebar
#                         url=payload.json()["data"]["screenshot"]["url"],
#                     )
#                 )

#         print(links)
#         msg_references.append(
#             cl.Message(
#                 content="\n".join([l.url for l in links]), elements=links
#             )
#         )

#     return streamed_text, msg_references
from langchain_core.documents import Document


def format_video_reference(doc: Document):
    v = {k: doc.metadata[k] for k in ("title", "source", "start", "stop")}

    v["start_min"] = f"{round(v['start'] // 60)}m:{round(v['start'] % 60)}s"
    video_link = cl.Video(
        name=v["title"],
        url=f"{v['source']}#t={v['start']}",
        display="side",
    )
    video_message = cl.Message(
        content=f'Watch {video_link.name} (_@ {v["start_min"]}_) for reference',  # text has to include video name
        elements=[video_link],
    )

    return video_message


def format_url_reference(url):
    microlink = "https://api.microlink.io"
    params = {
        "url": url,
        "screenshot": True,
    }

    payload = requests.get(microlink, params)

    screenshots = []
    if payload:
        print(f"Successful screenshot\n{payload.json()}")
        screenshots.append(
            cl.Image(
                name=f"Website Preview: {url}",
                display="side",  # Show in the sidebar
                url=payload.json()["data"]["screenshot"]["url"],
            )
        )

    print(links)
    msg_references.append(
        cl.Message(content="\n".join([l.url for l in links]), elements=links)
    )


from langchain.callbacks.base import BaseCallbackHandler


class ChainlitCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.current_step = None
        self.step_counter = 0

    async def on_chain_start(self, serialized, inputs, **kwargs):
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
            self.step_counter += 1

            print(f"Error in on_chain_start: {e}")
            self.current_step = cl.Step(
                name=f"Exception step_{self.step_counter}"
            )
            await self.current_step.__aenter__()

    async def on_chain_end(self, outputs, **kwargs):
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
                    pass

    async def on_chain_error(self, error, **kwargs):
        """Handle errors and close the step"""
        try:
            if self.current_step:
                self.current_step.output = f"Error: {str(error)}"
                await self.current_step.__aexit__(None, None, None)
                self.current_step = None
        except Exception as e:
            print(f"Error in on_chain_error: {e}")


@cl.on_message
async def main(input_message: cl.Message):
    """
    Processes incoming user messages and sends responses.

    Streams the AI agent's response, processes it to extract text and
    video references, and sends the content back to the user's chat interface.

    Args:
        message: User's input message
    """
    ai_graph = cast(Runnable, cl.user_session.get("ai_graph"))
    configuration = cl.user_session.get("configuration")

    if not configuration:
        await cl.Message(content="Error: Configuration not found").send()
        return

    # Convert Configuration to RunnableConfig format
    config = configuration.to_runnable_config()
    config["callbacks"] = [ChainlitCallbackHandler()]

    response = await ai_graph.ainvoke({"query": input_message.content}, config)

    for msg in response["messages"]:
        await cl.Message(content=msg.content, author=msg.type).send()

    for v in response["video_references"]:
        await format_video_reference(v).send()


if __name__ == "__main__":
    # This will start the Chainlit app when run directly
    pass
