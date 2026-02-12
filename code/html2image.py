"""
Simple HTML to Image Converter
Just provide HTML file name/path, get an image saved in the same location
"""

import os
import sys
import argparse
from playwright.sync_api import sync_playwright


def html_to_image(html_file_path):
    """
    Convert HTML file to image - simple and straightforward

    Args:
        html_file_path: Path to HTML file (e.g., "slide1.html" or "folder/slide1.html")
        width: Image width in pixels (default: 1280)
        height: Image height in pixels (default: 720)

    Returns:
        Path to generated image file
    """

    # Check if file exists
    if not os.path.exists(html_file_path):
        print(f"❌ HTML file not found: {html_file_path}")
        return None

    # Generate image path (same location, same name, but .png)
    image_path = os.path.splitext(html_file_path)[0] + ".png"

    try:
        # Render to image using Playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            page.goto("file:///" + os.path.abspath(html_file_path))
            page.wait_for_load_state("networkidle")

            # Set large viewport to avoid clipping
            page.set_viewport_size({"width": 10000, "height": 10000})

            # Get the complete position and size of the .poster area
            poster_box = page.query_selector(".poster").bounding_box()
            print(f"Poster box: {poster_box}")

            # Capture entire .poster area (using clip)
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

        # Display in terminal (show file info)
        file_size = os.path.getsize(image_path) / 1024  # KB
        print(f"📸 Image: {os.path.basename(image_path)} ({file_size:.1f} KB, {poster_box['width']}x{poster_box['height']})")

        return image_path

    except Exception as e:
        print(f"❌ Failed to convert {html_file_path}: {e}")
        return None


def process_html_folder(folder_path, output_dir=None, image_subfolder_name="images", width=1280, height=720):
    """
    Process all HTML files in a folder and save images to specified output directory

    Args:
        folder_path: Path to folder containing HTML files
        output_dir: Path to output directory (if None, creates subfolder in input folder)
        image_subfolder_name: Name of subfolder to create for images (default: "images")
    """

    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return

    # Find all HTML files in the folder
    html_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.html', '.htm'))]

    if not html_files:
        print(f"❌ No HTML files found in: {folder_path}")
        return

    # Determine output directory
    if output_dir:
        # Use specified output directory
        images_folder = output_dir
        os.makedirs(images_folder, exist_ok=True)
    else:
        # Create subfolder in input directory
        images_folder = os.path.join(folder_path, image_subfolder_name)
        os.makedirs(images_folder, exist_ok=True)

    print(f"📁 Found {len(html_files)} HTML files in: {folder_path}")
    print(f"📸 Images will be saved to: {images_folder}")
    print("-" * 50)

    # Process each HTML file
    processed = 0
    for i, html_file in enumerate(html_files, 1):
        html_path = os.path.join(folder_path, html_file)
        image_name = os.path.splitext(html_file)[0] + ".png"
        image_path = os.path.join(images_folder, image_name)

        print(f"🔄 [{i}/{len(html_files)}] Processing: {html_file}")

        try:
            # Read HTML content
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            # Render to image using Playwright
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.set_viewport_size({"width": width, "height": height})
                page.set_content(html_content)
                page.wait_for_load_state("networkidle")
                page.screenshot(path=image_path, full_page=False)
                browser.close()

            file_size = os.path.getsize(image_path) / 1024  # KB
            print(f"   ✅ Saved: {image_name} ({file_size:.1f} KB)")
            processed += 1

        except Exception as e:
            print(f"   ❌ Failed: {html_file} - {e}")

    print("-" * 50)
    print(f"🎉 Complete! Processed {processed}/{len(html_files)} files")
    print(f"📂 Images saved in: {images_folder}")


# Simple usage examples
# if __name__ == "__main__":
#     # Example 1: Just filename (if in same folder)
#     html_to_image("slide1.html")

#     # Example 2: Full path
#     html_to_image("C:/Users/Downloads/my_slide.html")

#     # Example 3: Custom size
#     html_to_image("slide2.html", width=1920, height=1080)

# Main execution with command line argument support
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🎯 HTML to Image Converter - Convert HTML files to PNG images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all HTML files in a folder (creates 'images' subfolder)
  python htm2image.py --input_dir htmls

  # Process with custom output directory
  python htm2image.py --input_dir htmls --output_dir ../output_images

  # Custom image dimensions
  python htm2image.py --input_dir htmls --width 1920 --height 1080

  # Convert single HTML file
  python htm2image.py slide1.html

  # Default: process htmls folder if no arguments
  python htm2image.py
        """
    )

    parser.add_argument('file_or_dir', nargs='?',
                       help='Single HTML file or directory path (if not using --input_dir)')
    parser.add_argument('--input_dir', '-i',
                       help='Directory containing HTML files to process')
    parser.add_argument('--output_dir', '-o',
                       help='Output directory for images (default: creates "images" subfolder in input dir)')
    parser.add_argument('--width', type=int, default=1280,
                       help='Image width in pixels (default: 1280)')
    parser.add_argument('--height', type=int, default=720,
                       help='Image height in pixels (default: 720)')

    args = parser.parse_args()

    print("🎯 HTML to Image Converter")
    print("=" * 50)

    # Determine input source
    if args.input_dir:
        # Directory processing mode
        input_dir = args.input_dir
        print(f"📁 Input directory: {input_dir}")
        if args.output_dir:
            print(f"📸 Output directory: {args.output_dir}")
        else:
            print(f"📸 Output: Creating 'images' subfolder in input directory")
        # print(f"🖼️  Image size: {args.width}x{args.height}")
        print("-" * 50)
        process_html_folder(input_dir, output_dir=args.output_dir, width=args.width, height=args.height)

    elif args.file_or_dir:
        # Check if it's a file or directory
        if os.path.isfile(args.file_or_dir) and args.file_or_dir.lower().endswith(('.html', '.htm')):
            # Single file mode
            print(f"📄 Processing single file: {args.file_or_dir}")
            print(f"🖼️  Image size: {args.width}x{args.height}")
            print("-" * 50)
            html_to_image(args.file_or_dir)
        elif os.path.isdir(args.file_or_dir):
            # Directory mode (positional argument)
            print(f"📁 Input directory: {args.file_or_dir}")
            if args.output_dir:
                print(f"📸 Output directory: {args.output_dir}")
            else:
                print(f"📸 Output: Creating 'images' subfolder in input directory")
            print(f"🖼️  Image size: {args.width}x{args.height}")
            print("-" * 50)
            process_html_folder(args.file_or_dir, output_dir=args.output_dir, width=args.width, height=args.height)
        else:
            print(f"❌ Invalid input: {args.file_or_dir} (not an HTML file or directory)")
            sys.exit(1)

    else:
        # Default mode - use htmls folder
        default_dir = "htmls"
        if os.path.exists(default_dir):
            print(f"📁 Using default directory: {default_dir}")
            print(f"📸 Output: Creating 'images' subfolder")
            print(f"🖼️  Image size: {args.width}x{args.height}")
            print("-" * 50)
            process_html_folder(default_dir, width=args.width, height=args.height)
        else:
            print("❌ No input specified and default 'htmls' folder not found.")
            print("Use: python htm2image.py --help for usage information")
            sys.exit(1)