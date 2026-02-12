"""
HTML to Image Converter using Playwright
"""

import os
from playwright.sync_api import sync_playwright


def html_to_image(html_file_path):
    """
    Convert HTML file to image — captures the .poster element.

    Args:
        html_file_path: Path to HTML file

    Returns:
        Path to generated PNG, or None on failure
    """
    if not os.path.exists(html_file_path):
        print(f"❌ HTML file not found: {html_file_path}")
        return None

    image_path = os.path.splitext(html_file_path)[0] + ".png"

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto("file:///" + os.path.abspath(html_file_path))
            page.wait_for_load_state("networkidle")
            page.set_viewport_size({"width": 10000, "height": 10000})

            poster_box = page.query_selector(".poster").bounding_box()
            page.screenshot(
                path=image_path,
                clip={
                    "x": poster_box["x"],
                    "y": poster_box["y"],
                    "width": poster_box["width"],
                    "height": poster_box["height"],
                }
            )
            browser.close()

        print(f"✅ Image saved: {image_path}")
        return image_path

    except Exception as e:
        print(f"❌ Failed to convert {html_file_path}: {e}")
        return None