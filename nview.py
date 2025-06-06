import asyncio
import pandas as pd
from playwright.async_api import async_playwright

CHROME_PATH = "C:\Program Files\Google\Chrome\Application\chrome.exe"

def safe_strip(value):
    return value.strip() if value else ""

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
        print(f"[ERROR] Failed to get views for @{username}: {e}")
    finally:
        await page.close()
    return ""

async def main():
    df = pd.read_csv("tiktok_users.csv")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            slow_mo=50,
            executable_path=CHROME_PATH
        )
        context = await browser.new_context()

        for idx, row in df.iterrows():
            if pd.isna(row["vid_nview"]) or str(row["vid_nview"]).strip() == "":
                username = row["user_name"]
                video_id = str(row["vid_id"])
                print(f"\n[{idx + 1}] Refetching views for @{username} - {video_id}")
                try:
                    views = await get_video_views(context, username, video_id)
                    df.at[idx, "vid_nview"] = views
                except Exception as e:
                    print(f"    [!] Error: {e}")

        await browser.close()

    df.to_csv("users.csv", index=False, encoding="utf-8-sig")
    print("\nView data updated and saved.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Fatal error: {e}")
