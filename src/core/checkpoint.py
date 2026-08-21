import os
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

checkpoint = None
checkpoint_cm = None


async def init_checkpoint():

    global checkpoint
    global checkpoint_cm

    CHECKPOINT_DB_URI = os.getenv("LANGGRAPH_CHECKPOINT_DB_URI")

    checkpoint_cm = AsyncPostgresSaver.from_conn_string(CHECKPOINT_DB_URI)

    checkpoint = await checkpoint_cm.__aenter__()

    await checkpoint.setup()

    return checkpoint


async def close_checkpoint():

    global checkpoint_cm

    if checkpoint_cm:
        await checkpoint_cm.__aexit__(None, None, None)
