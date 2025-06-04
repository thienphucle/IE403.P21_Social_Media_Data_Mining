import asyncio
import csv
import time
import re
import datetime
import pandas as pd
from playwright.async_api import async_playwright

CHROME_PATH = "C:\Program Files\Google\Chrome\Application\chrome.exe"

def safe_strip(value):
    return value.strip() if value else ""

def get_post_time(video_id: str) -> str:
    try:
        video_id_int = int(video_id)
        binary = format(video_id_int, '064b')
        timestamp_bin = binary[:32]
        timestamp_int = int(timestamp_bin, 2)
        post_time = datetime.datetime.utcfromtimestamp(timestamp_int)
        return post_time.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        return f"Error: {e}"

async def get_user_followers(context, username):
    page = await context.new_page()
    try:
        await page.goto(f"https://www.tiktok.com/@{username}", timeout=30000)
        await page.wait_for_selector('strong[data-e2e="followers-count"]', timeout=20000)
        follower_elem = await page.query_selector('strong[data-e2e="followers-count"]')
        if follower_elem:
            followers = safe_strip(await follower_elem.text_content())
            return followers
    except Exception as e:
        print(f"Error getting followers for @{username}: {e}")
    finally:
        await page.close()
    return ""

async def get_video_views(context, username, target_video_id):
    page = await context.new_page()
    try:
        await page.goto(f"https://www.tiktok.com/@{username}", timeout=30000)
        await page.wait_for_selector('div[data-e2e="user-post-item"]', timeout=20000)
        
        video_cards = await page.query_selector_all('div[data-e2e="user-post-item"]')
        for card in video_cards:
            link_tag = await card.query_selector("a")
            video_url = await link_tag.get_attribute("href") if link_tag else ""
            if target_video_id in video_url:
                views_elem = await card.query_selector('strong[data-e2e="video-views"]')
                views = safe_strip(await views_elem.text_content()) if views_elem else ""
                return views
    except Exception as e:
        print(f"Could not get views for @{username}: {e}")
    finally:
        await page.close()
    return ""

async def get_music_info(context, music_href, username):
    sound_id = ""
    sound_title = ""
    uses_sound_count = ""
    music_author = ""
    music_originality = ""

    if not music_href:
        return sound_id, sound_title, uses_sound_count, music_author, music_originality

    try:
        href = await music_href.get_attribute("href")
        if href:
            music_url = f"https://www.tiktok.com{href}"
            sound_page = await context.new_page()
            await sound_page.goto(music_url, timeout=20000)
            await sound_page.wait_for_timeout(2000)

            match = re.search(r'(\d{10,})/?$', href)
            if match:
                sound_id = match.group(1)

            music_title_elem = await sound_page.query_selector('h1[data-e2e="music-title"]')
            if music_title_elem:
                sound_title = safe_strip(await music_title_elem.text_content())

            uses_elem = await sound_page.query_selector('h2[data-e2e="music-video-count"] strong')
            if uses_elem:
                uses_sound_count = safe_strip(await uses_elem.text_content())

            music_author_elems = await sound_page.query_selector_all('h2[data-e2e="music-creator"] a')
            music_authors = []
            music_usernames = []

            for author_elem in music_author_elems:
                display_name = safe_strip(await author_elem.text_content())
                author_href = await author_elem.get_attribute("href")
                if author_href and author_href.startswith("/@"):
                    username_only = author_href.split("/")[-1].lstrip("@").lower()
                    music_usernames.append(username_only)
                music_authors.append(display_name)

            music_author = "|".join(music_authors)
            music_originality = "true" if username.lower() in music_usernames else "false"

            await sound_page.close()

    except Exception as e:
        print(f"Error extracting music info: {e}")

    return sound_id, sound_title, uses_sound_count, music_author, music_originality

async def extract_from_video_page(context, video_url, username, pre_fetched_views=""):
    page = await context.new_page()
    try:
        await page.goto(video_url, timeout=30000)
        await page.wait_for_selector('strong[data-e2e="like-count"]', timeout=10000)

        video_id = video_url.split("/")[-1]
        # Use the pre-fetched views if available; fallback to get_video_views
        views = pre_fetched_views or await get_video_views(context, username, video_id)
        post_time = get_post_time(video_id)
        followers = await get_user_followers(context, username)

        caption_elem = await page.query_selector('span[data-e2e="new-desc-span"]')
        likes_elem = await page.query_selector('strong[data-e2e="like-count"]')
        comments_elem = await page.query_selector('strong[data-e2e="comment-count"]')
        shares_elem = await page.query_selector('strong[data-e2e="share-count"]')
        saves_elem = await page.query_selector('strong[data-e2e="undefined-count"]')

        hashtags = []
        hashtag_links = await page.query_selector_all('a[data-e2e="search-common-link"]')
        for tag in hashtag_links:
            href = await tag.get_attribute("href")
            if href and "/tag/" in href:
                hashtags.append(href.split("/tag/")[-1])

        duration = ""
        duration_elem = await page.query_selector('div[class*="DivSeekBarTimeContainer"]')
        if duration_elem:
            duration_text = await duration_elem.text_content()
            duration = duration_text.split('/')[-1].strip() if "/" in duration_text else duration_text.strip()

        # Music info
        music_href = await page.query_selector('a[data-e2e="video-music"]') or \
                    await page.query_selector('h4[data-e2e="video-music"] a') or \
                    await page.query_selector('h4[data-e2e="browse-music"] a')

        sound_id = sound_title = uses_sound_count = music_author = music_originality = ""

        if music_href:
            href = await music_href.get_attribute("href")
            if href:
                music_url = f"https://www.tiktok.com{href}"
                sound_page = await context.new_page()
                await sound_page.goto(music_url, timeout=20000)
                await sound_page.wait_for_timeout(2000)

                try:
                    match = re.search(r'(\d{10,})/?$', href)
                    if match:
                        sound_id = match.group(1)

                    music_title_elem = await sound_page.query_selector('h1[data-e2e="music-title"]')
                    if music_title_elem:
                        sound_title = safe_strip(await music_title_elem.text_content())

                    uses_elem = await sound_page.query_selector('h2[data-e2e="music-video-count"] strong')
                    if uses_elem:
                        uses_sound_count = safe_strip(await uses_elem.text_content())

                    music_author_elems = await sound_page.query_selector_all('h2[data-e2e="music-creator"] a')
                    music_authors = []
                    music_usernames = []

                    for author_elem in music_author_elems:
                        display_name = safe_strip(await author_elem.text_content())
                        author_href = await author_elem.get_attribute("href")
                        if author_href and author_href.startswith("/@"):
                            username_only = author_href.split("/")[-1].lstrip("@").lower()
                            music_usernames.append(username_only)
                        music_authors.append(display_name)

                    music_author = "|".join(music_authors)
                    music_originality = "true" if username.lower() in music_usernames else "false"

                except Exception as e:
                    print("Lỗi khi trích xuất thông tin âm thanh:", e)
                await sound_page.close()

        return {
            "user_name": username,
            "user_nfollower": followers,
            "vid_id": video_id,
            "vid_caption": safe_strip(await caption_elem.text_content() if caption_elem else ""),
            "vid_postTime": post_time,
            "vid_scrapeTime": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "vid_duration": duration,
            "vid_nview": views,
            "vid_nlike": safe_strip(await likes_elem.text_content() if likes_elem else ""),
            "vid_ncomment": safe_strip(await comments_elem.text_content() if comments_elem else ""),
            "vid_nshare": safe_strip(await shares_elem.text_content() if shares_elem else ""),
            "vid_nsave": safe_strip(await saves_elem.text_content() if saves_elem else ""),
            "vid_hashtags": ", ".join(hashtags),
            "vid_url": video_url,
            "music_id": sound_id,
            "music_title": sound_title,
            "music_nused": uses_sound_count,
            "music_authorName": music_author,
            "music_originality": music_originality
        }

    except Exception as e:
        print(f"Error scraping video page {video_url}: {e}")
        return None
    finally:
        await page.close()

async def collect_user_videos(context, username, viral_video_id, num_videos=10):
    page = await context.new_page()
    enriched_info = []
    seen_urls = set()
    results = []

    try:
        await page.goto(f"https://www.tiktok.com/@{username}", timeout=60000)
        print(f"Scraping videos for @{username}...")

        await page.wait_for_selector('div[data-e2e="user-post-item-list"]', timeout=20000)

        while True:
            cards = await page.query_selector_all('div[data-e2e="user-post-item"]')
            new_found = False

            for card in cards:
                link_tag = await card.query_selector("a")
                video_url = await link_tag.get_attribute("href") if link_tag else ""
                if video_url and video_url not in seen_urls:
                    seen_urls.add(video_url)
                    vid_id = video_url.split("/")[-1]
                    post_time = get_post_time(vid_id)

                    enriched_info.append({
                        "video_url": video_url,
                        "video_id": vid_id,
                        "post_time": post_time,
                        "card": card
                    })
                    new_found = True

            video_ids = [v["video_id"] for v in enriched_info]

            if viral_video_id in video_ids:
                viral_info = next(v for v in enriched_info if v["video_id"] == viral_video_id)
                viral_card = viral_info["card"]
                is_pinned = await viral_card.query_selector('div[class*="DivHeaderContainer"]') is not None

                if is_pinned:
                    enriched_info.sort(key=lambda x: datetime.datetime.strptime(x["post_time"], "%Y-%m-%d %H:%M:%S"))
                    sorted_ids = [v["video_id"] for v in enriched_info]
                    index = sorted_ids.index(viral_video_id)
                else:
                    index = video_ids.index(viral_video_id)

                enough_before = index >= (num_videos + 2)
                enough_after = (len(enriched_info) - index - 1) >= (num_videos + 2)

                if enough_before and enough_after:
                    break

            if not new_found:
                break

            await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
            await page.wait_for_timeout(1500)

        if viral_video_id not in [v["video_id"] for v in enriched_info]:
            print("Viral video not found in enriched info.")
            return []

        if is_pinned:
            enriched_info.sort(key=lambda x: datetime.datetime.strptime(x["post_time"], "%Y-%m-%d %H:%M:%S"))
            index = [v["video_id"] for v in enriched_info].index(viral_video_id)
        else:
            index = [v["video_id"] for v in enriched_info].index(viral_video_id)

        start = max(0, index - num_videos)
        end = min(len(enriched_info), index + num_videos + 1)

        related = enriched_info[start:index] + enriched_info[index+1:end]

        for video in related:
            print(f"Extracting video: {video['video_url']}")
            views = ""
            metadata = await extract_from_video_page(context, video['video_url'], username, pre_fetched_views=views)
            if metadata:
                results.append(metadata)

    except Exception as e:
        print(f"Error scraping user videos: {e}")
    finally:
        await page.close()

    return results


def save_to_csv(data, filename='all_users_videos.csv'):
    if not data:
        print("No data to save")
        return

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"Saved {len(data)} videos to {filename}")

async def main():
    input_csv = "test.csv"
    df = pd.read_csv(input_csv)

    all_results = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            slow_mo=50,
            executable_path=CHROME_PATH
        )
        context = await browser.new_context()

        for idx, row in df.iterrows():
            username = row['user_name']
            viral_video_id = str(row['vid_id'])
            print(f"\nCrawling videos for @{username} (viral_id: {viral_video_id})")

            try:
                videos = await collect_user_videos(context, username, viral_video_id)
                all_results.extend(videos)
            except Exception as e:
                print(f"Failed for @{username}: {e}")

        await browser.close()

    save_to_csv(all_results, "all_users_videos.csv")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Fatal error: {e}")