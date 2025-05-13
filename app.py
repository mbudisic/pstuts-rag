import chainlit as cl

@cl.on_message
async def main(message: cl.Message):
    # Send a response back to the user
    await cl.Message(
        content=f"Hello! You said: {message.content}"
    ).send()


if __name__ == "__main__":
    main()
