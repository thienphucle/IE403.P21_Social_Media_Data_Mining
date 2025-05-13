import asyncio
import csv
import time
from playwright.async_api import async_playwright

CHROME_PATH = "C:\Program Files\Google\Chrome\Application\chrome.exe"


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
                def safe_strip(value):
                    return value.strip() if value else ""

                # Crawl info
                username = await current_video.query_selector('h3[data-e2e="video-author-uniqueid"]')
                likes = await current_video.query_selector('strong[data-e2e="like-count"]')
                comments = await current_video.query_selector('strong[data-e2e="comment-count"]')
                shares = await current_video.query_selector('strong[data-e2e="share-count"]')
                undefined = await current_video.query_selector('strong[data-e2e="undefined-count"]')
                caption_elem = await current_video.query_selector('span[data-e2e="new-desc-span"]')

                username = safe_strip(await username.text_content() if username else "")
                likes = safe_strip(await likes.text_content() if likes else "")
                comments = safe_strip(await comments.text_content() if comments else "")
                shares = safe_strip(await shares.text_content() if shares else "")
                undefined = safe_strip(await undefined.text_content() if undefined else "")
                caption = safe_strip(await caption_elem.text_content() if caption_elem else "")

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

                # Âm thanh
                sound_id = ""
                sound_title = ""
                uses_sound_count = ""

                music_href = await current_video.query_selector('a[data-e2e="video-music"]')
                if music_href:
                    href = await music_href.get_attribute("href")
                    if href:
                        music_url = f"https://www.tiktok.com{href}"
                        sound_page = await context.new_page()
                        await sound_page.goto(music_url, timeout=20000)
                        await sound_page.wait_for_timeout(2000)

                        try:
                            # ID âm thanh từ URL
                            if "music" in href:
                                sound_id = href.split("-")[-1]

                            # Tiêu đề nhạc
                            music_title_elem = await sound_page.query_selector('h1[data-e2e="music-title"]')
                            if music_title_elem:
                                sound_title = safe_strip(await music_title_elem.text_content())

                            # Lượt sử dụng nhạc
                            uses_elem = await sound_page.query_selector('h2[data-e2e="music-video-count"] strong')
                            if uses_elem:
                                uses_sound_count = safe_strip(await uses_elem.text_content())
                        except Exception as e:
                            print("Lỗi khi trích xuất thông tin âm thanh:", e)

                        await sound_page.close()

                results.append({
                    "username": username,
                    "caption": caption,
                    "likes": likes,
                    "comments": comments,
                    "shares": shares,
                    "undefined": undefined,
                    "hashtags": ", ".join(hashtags),
                    "sound_id": sound_id,
                    "sound_title": sound_title,
                    "uses_sound_count": uses_sound_count,
                    "vid_id": safe_strip(video_id),
                    "vid_url": safe_strip(video_url),
                })

                print(f"Đã thu thập video #{i+1} của @{username}")

            except Exception as e:
                print(f"Bỏ qua video #{i+1} do lỗi:", e)

            # scrollIntoView
            try:
                next_index = i + 1
                next_selector = f'article[data-scroll-index="{next_index}"]'
                next_video = await page.query_selector(next_selector)

                if next_video:
                    await next_video.scroll_into_view_if_needed()
                    await page.wait_for_timeout(3000)
                else:
                    print(f"Không tìm thấy video tiếp theo với data-scroll-index={next_index}")
            except Exception as e:
                print("Lỗi khi cuộn đến video tiếp theo:", e)

        await browser.close()
        end_time = time.time()
        print(f"\nTổng thời gian:  {round(end_time - start_time, 2)} giây.")
        return results


def save_to_csv(data, filename="tiktok_feed.csv"):
    with open(filename, mode="w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "username", "caption", "likes", "comments", "shares", "undefined", 
            "hashtags", "sound_id", "sound_title", "uses_sound_count", "video_id", "video_url"
        ])
        writer.writeheader()
        writer.writerows(data)
    print(f"Đã lưu {len(data)} video vào file {filename}")


if __name__ == "__main__":
    data = asyncio.run(scrape_feed(num_scrolls=3))
    save_to_csv(data)




