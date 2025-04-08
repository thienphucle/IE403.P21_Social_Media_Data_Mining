import asyncio
import json
from playwright.async_api import async_playwright

CHROME_PATH = "C:\Program Files\Google\Chrome\Application\chrome.exe"

async def save_cookies():
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            slow_mo=50,
            executable_path=CHROME_PATH
        )
        context = await browser.new_context()
        page = await context.new_page()

        print("Đang mở TikTok để bạn đăng nhập...")
        await page.goto("https://www.tiktok.com/login", timeout=60000)

        input("Nhấn Enter để lưu cookies...")

        cookies = await context.cookies()
        with open("cookies.json", "w", encoding="utf-8") as f:
            json.dump(cookies, f, ensure_ascii=False, indent=2)
        print("Cookies đã được lưu vào file cookies.json")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(save_cookies())
