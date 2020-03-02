import aiohttp
import asyncio
import async_timeout
import datetime as dt
import json
import requests
import bs4
import functools
from multiprocessing import Pool
import subprocess


def build_url(date):
    BASE_URL = 'https://www.rtp.pt/EPG/json/rtp-home-page-tv-radio/list-all-grids/tv/'
    url = BASE_URL + f'{date.day}-{date.month}-{date.year}'
    return url


def weekdays_range(start, end):
    SATURDAY, SUNDAY = 5, 6
    day = start
    while day.weekday() in [SATURDAY, SUNDAY]:
        day += dt.timedelta(days=1)

    while day <= end:
        yield day
        day += dt.timedelta(days=1)
        while day.weekday() in [SATURDAY, SUNDAY]:
            day += dt.timedelta(days=1)


async def aired_on_date(loop, date):
    print(date)
    PRICE_IS_RIGHT_STRING = 'O Preço Certo'

    url = build_url(date)
    async with aiohttp.ClientSession(loop=loop) as session:
        async with session.get(url) as response:
            afternoon_programming = json.loads(await response.text())['result']['rtp-1']['afternoon']
            for show in afternoon_programming:
                if show['name'] == PRICE_IS_RIGHT_STRING:
                    return True

            return False


async def task_aired_on_date(sem, loop, date):
    async with sem:
        was_aired = await aired_on_date(loop, date)
    if was_aired:
        return str(date)
    else:
        return None


async def dump_aired_days(loop, start, end):
    sem = asyncio.BoundedSemaphore(10)
    tasks = [
        asyncio.create_task(
            task_aired_on_date(sem, loop, date))
        for date in weekdays_range(start, end)
    ]

    results = await asyncio.gather(*tasks, loop=loop)

    print(results)
    return results


def main():
    start = dt.date(2016, 3, 30)
    end = dt.date(2020, 2, 28)

    loop = asyncio.get_event_loop()
    aired_dates = loop.run_until_complete(dump_aired_days(loop, start, end))
    with open('./data/aired_dates.txt', 'w') as filehandle:
        filehandle.writelines(f"{date}\n" for date in aired_dates if date)


main()
