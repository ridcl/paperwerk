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

    async def _gather():
        if limit is None:
            return await asyncio.gather(*coros)
        return await gather_limited(coros, limit)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_gather())
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return pool.submit(asyncio.run, _gather()).result()
