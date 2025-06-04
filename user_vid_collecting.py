import asyncio
import csv
import time
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

async def extract_video_data(video_card, context, username):
    try:
        username_elem = await video_card.query_selector('h3[data-e2e="video-author-uniqueid"]')
        likes_elem = await video_card.query_selector('strong[data-e2e="like-count"]')
        comments_elem = await video_card.query_selector('strong[data-e2e="comment-count"]')
        shares_elem = await video_card.query_selector('strong[data-e2e="share-count"]')
        undefined_elem = await video_card.query_selector('strong[data-e2e="undefined-count"]')
        caption_elem = await video_card.query_selector('span[data-e2e="new-desc-span"]')

        link_tag = await video_card.query_selector("a")
        video_url = await link_tag.get_attribute("href") if link_tag else ""
        video_id = video_url.split("/")[-1] if video_url else ""

        views = await get_video_views(context, username, video_id)
        followers = await get_user_followers(context, username)
        post_time = get_post_time(video_id)

        duration = ""
        duration_elem = await video_card.query_selector('p[class*="StyledTimeDisplayText"]')
        if duration_elem:
            duration_text = await duration_elem.text_content()
            duration = duration_text.split('/')[-1].strip() if "/" in duration_text else duration_text.strip()

        hashtags = []
        hashtag_links = await video_card.query_selector_all('a[data-e2e="search-common-link"]')
        for tag in hashtag_links:
            href = await tag.get_attribute("href")
            if href and "/tag/" in href:
                hashtags.append(href.split("/tag/")[-1])

        sound_id = ""
        sound_title = ""
        uses_sound_count = ""
        music_author = ""
        music_originality = ""

        music_href = await video_card.query_selector('a[data-e2e="video-music"]') or \
                     await video_card.query_selector('h4[data-e2e="video-music"] a') or \
                     await video_card.query_selector('h4[data-e2e="browse-music"] a')

        if music_href:
            href = await music_href.get_attribute("href")
            if href:
                music_url = f"https://www.tiktok.com{href}"
                sound_page = await context.new_page()
                await sound_page.goto(music_url, timeout=20000)
                await sound_page.wait_for_timeout(2000)

                try:
                    import re
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
                    print(f"Error extracting music info: {e}")
                await sound_page.close()

        else:
            music_text_div = await video_card.query_selector('div[class*="DivMusicText"]')
            if music_text_div:
                music_text = await music_text_div.text_content()
                if music_text:
                    sound_title = safe_strip(music_text)

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
            "vid_nsave": safe_strip(await undefined_elem.text_content() if undefined_elem else ""),
            "vid_hashtags": ", ".join(hashtags),
            "vid_url": video_url,
            "music_id": sound_id,
            "music_title": sound_title,
            "music_nused": uses_sound_count,
            "music_authorName": music_author,
            "music_originality": music_originality
        }
    except Exception as e:
        print(f"Error extracting video data: {e}")
        return None


async def collect_user_videos(context, username, viral_video_id, num_videos=3):
    page = await context.new_page()
    results = []

    try:
        await page.goto(f"https://www.tiktok.com/@{username}", timeout=60000)
        print(f"Loading videos for @{username}...")
        await page.wait_for_selector('div[data-e2e="user-post-item-list"]', timeout=20000)

        found_viral = False
        collected_before = 0
        collected_after = 0
        seen_ids = set()

        while True:
            video_cards = await page.query_selector_all('div[data-e2e="user-post-item-list"]')
            new_cards = [card for card in video_cards if card not in seen_ids]
            for card in new_cards:
                link_tag = await card.query_selector("a")
                video_url = await link_tag.get_attribute("href") if link_tag else ""
                if not video_url:
                    continue

                current_video_id = video_url.split("/")[-1]

                seen_ids.add(card)

                if current_video_id == viral_video_id:
                    found_viral = True
                    continue

                if found_viral and collected_after < num_videos:
                    video_data = await extract_video_data(card, context, username)
                    if video_data:
                        results.append(video_data)
                        collected_after += 1
                elif not found_viral and collected_before < num_videos:
                    video_data = await extract_video_data(card, context, username)
                    if video_data:
                        results.insert(0, video_data)
                        collected_before += 1

                if collected_before >= num_videos and collected_after >= num_videos:
                    break

            if collected_before >= num_videos and collected_after >= num_videos:
                break

            await page.evaluate("window.scrollBy(0, 1000)")
            await page.wait_for_timeout(2000)

    except Exception as e:
        print(f"Error collecting videos for @{username}: {e}")
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
    input_csv = "test7.csv"
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
