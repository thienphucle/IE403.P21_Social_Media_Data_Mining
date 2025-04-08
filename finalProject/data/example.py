from TikTokApi import TikTokApi
import asyncio
import os

ms_token = 'owuxBlOj6Vs0HtIVZCvMHVokmeF1AXll4coL2UjRiwSOSq7Wh6_gczNCTsHW3nN8LsnM077N9YNdNa3Ca6NDMAMMETktl-1LwDZB-FxNQMgfy-gLU4-gjzaOdiqDKQkL7iW828gMdSCYwANj3jS33i_G'
ms_token = os.environ.get("ms_token", None) # get your own ms_token from your cookies on tiktok.com

async def trending_videos():
    async with TikTokApi() as api:
        await api.create_sessions(ms_tokens=[ms_token], num_sessions=1, sleep_after=3, browser=os.getenv("TIKTOK_BROWSER", "chromium"))
        async for video in api.trending.videos(count=30):
            print(video)
            print(video.as_dict)

if __name__ == "__main__":
    asyncio.run(trending_videos())