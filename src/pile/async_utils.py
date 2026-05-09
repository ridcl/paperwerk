import asyncio
import concurrent.futures


async def gather_limited(coros, limit):
    semaphore = asyncio.Semaphore(limit)

    async def sem_coro(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(sem_coro(c) for c in coros))


def run_async(coros, limit: int = None):
    coros = list(coros)  # consume generator once
    gathered = (
        asyncio.gather(*coros) if limit is None else gather_limited(coros, limit)
    )
    try:
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, gathered).result()
    except RuntimeError:
        return asyncio.run(gathered)
