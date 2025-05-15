import asyncio
import csv
import time
import re
import datetime
from playwright.async_api import async_playwright

CHROME_PATH = "C:\Program Files\Google\Chrome\Application\chrome.exe"

def safe_strip(value):
    return value.strip() if value else ""

def get_post_time(video_id: str) -> str:
    try:
        video_id_int = int(video_id)
        binary = format(video_id_int, '064b')  # ensure 64-bit
        timestamp_bin = binary[:32]  # first 32 bits
        timestamp_int = int(timestamp_bin, 2)
        post_time = datetime.datetime.utcfromtimestamp(timestamp_int)
        return post_time.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        return f"Lỗi: {e}"

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
        print(f"Lỗi khi lấy số follower của @{username}: {e}")
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
        print(f"Không thể lấy view từ @{username}: {e}")
    finally:
        await page.close() 
    return ""

# Main crawler
async def scrape_feed(num_scrolls=10):
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            slow_mo=50,
            executable_path=CHROME_PATH
        )
        context = await browser.new_context()
        page = await context.new_page()

        await page.goto("https://www.tiktok.com/vi-VN/", timeout=60000)
        print("Đang chờ feed load...")
        await page.wait_for_selector('article[data-e2e="recommend-list-item-container"]', timeout=20000)

        results = []
        start_time = time.time()

        for i in range(num_scrolls):
            print(f"\nScroll #{i+1}")
            await page.wait_for_timeout(2000)

            video_selector = f'article[data-e2e="recommend-list-item-container"][data-scroll-index="{i}"]'
            current_video = await page.query_selector(video_selector)

            if not current_video:
                print(f"Không tìm thấy video với data-scroll-index = {i}")
                break

            try:
                scrape_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                username_elem = await current_video.query_selector('h3[data-e2e="video-author-uniqueid"]')
                likes_elem = await current_video.query_selector('strong[data-e2e="like-count"]')
                comments_elem = await current_video.query_selector('strong[data-e2e="comment-count"]')
                shares_elem = await current_video.query_selector('strong[data-e2e="share-count"]')
                undefined_elem = await current_video.query_selector('strong[data-e2e="undefined-count"]')
                caption_elem = await current_video.query_selector('span[data-e2e="new-desc-span"]')
                
                username = safe_strip(await username_elem.text_content() if username_elem else "")
                likes = safe_strip(await likes_elem.text_content() if likes_elem else "")
                comments = safe_strip(await comments_elem.text_content() if comments_elem else "")
                shares = safe_strip(await shares_elem.text_content() if shares_elem else "")
                undefined = safe_strip(await undefined_elem.text_content() if undefined_elem else "")
                caption = safe_strip(await caption_elem.text_content() if caption_elem else "")

                duration = ""
                duration_elem = await current_video.query_selector('p[class*="StyledTimeDisplayText"]')
                if duration_elem:
                    duration_text = await duration_elem.text_content()
                    if "/" in duration_text:
                        duration = duration_text.split('/')[-1].strip()
                    else:
                        duration = duration_text.strip()
                
                hashtags = []
                hashtag_links = await current_video.query_selector_all('a[data-e2e="search-common-link"]')
                for tag in hashtag_links:
                    href = await tag.get_attribute("href")
                    if href and "/tag/" in href:
                        hashtags.append(href.split("/tag/")[-1])
                
                video_id = None
                video_wrapper = await page.query_selector('div.tiktok-web-player')
                if video_wrapper:
                    wrapper_id = await video_wrapper.get_attribute('id')
                    if wrapper_id:
                        video_id = wrapper_id.split('-')[-1]
                video_url = f"https://www.tiktok.com/@{username}/video/{video_id}" if username and video_id else ""

                views = await get_video_views(context, username, video_id)
                followers = await get_user_followers(context, username)
                post_time = get_post_time(video_id)
                

                sound_id = ""
                sound_title = ""
                uses_sound_count = ""
                music_author = ""
                music_originality = ""

                music_href = await current_video.query_selector('a[data-e2e="video-music"]') or \
                                      await current_video.query_selector('h4[data-e2e="video-music"] a') or \
                                      await current_video.query_selector('h4[data-e2e="browse-music"] a')
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
                                href = await author_elem.get_attribute("href")
                                if href and href.startswith("/@"):
                                    username_only = href.split("/")[-1].lstrip("@").lower()
                                    music_usernames.append(username_only)
                                music_authors.append(display_name)

                            music_author = "|".join(music_authors)
                            music_originality = "true" if username.lower() in music_usernames else "false"

                        except Exception as e:
                            print("Lỗi khi trích xuất thông tin âm thanh:", e)
                        await sound_page.close()
                else:
                    music_text_div = await current_video.query_selector('div[class*="DivMusicText"]')
                    if music_text_div:
                        music_text = await music_text_div.text_content()
                        if music_text:
                            sound_title = safe_strip(music_text)

                # Add to results
                results.append({
                    "user_name": username,
                    "user_followers": followers,
                    "vid_id": safe_strip(video_id),
                    "vid_caption": caption,
                    "vid_postTime": post_time,
                    "vid_scrapeTime": scrape_time,
                    "vid_duration": duration,
                    "vid_nview": views,
                    "vid_nlike": likes,
                    "vid_ncomment": comments,
                    "vid_nshare": shares,
                    "vid_nsave": undefined,
                    "vid_hashtags": ", ".join(hashtags),
                    "vid_url": safe_strip(video_url),
                    "music_id": sound_id,
                    "music_title": sound_title,
                    "music_nused": uses_sound_count,
                    "music_authorName": music_author,
                    "music_originality":  music_originality,
                })
                print(f"Đã thu thập video #{i+1} của @{username}")

            except Exception as e:
                print(f"Bỏ qua video #{i+1} do lỗi:", e)

            try:
                next_index = i + 1
                next_selector = f'article[data-scroll-index="{next_index}"]'
                next_video = await page.query_selector(next_selector)

                if next_video:
                    await next_video.scroll_into_view_if_needed()
                    await page.wait_for_timeout(3000)
                else:
                    print(f"Không tìm thấy video tiếp theo {next_index}")
            except Exception as e:
                print("Lỗi khi cuộn đến video tiếp theo:", e)

        await browser.close()
        end_time = time.time()
        print(f"\nTổng thời gian:  {round(end_time - start_time, 2)} giây.")
        return results

def save_to_csv(data, filename='finalProject/data/tiktok_feed_1.csv'):
    with open(filename, mode="w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "user_name", "user_followers", "vid_id", "vid_caption", "vid_postTime", "vid_scrapeTime", 
            "vid_duration", "vid_nview", "vid_nlike", "vid_ncomment", "vid_nshare", "vid_nsave",
            "vid_hashtags", "vid_url", "music_id", "music_title", "music_nused", "music_authorName", "music_originality"
        ])
        writer.writeheader()
        writer.writerows(data)
    print(f"\nĐã lưu {len(data)} video vào file {filename}")

if __name__ == "__main__":
    data = asyncio.run(scrape_feed(num_scrolls=400))
    save_to_csv(data)
