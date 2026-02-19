"""
HTMLtoPNG
---------
Converts a local HTML file to a PNG screenshot using Playwright.

Requirements:
    pip install playwright
    playwright install chromium
"""

import asyncio
from pathlib import Path
from playwright.async_api import async_playwright
import json


class HTMLtoPNG:
    def __init__(
        self,
        output_dir: str = "png_outputs",
        width: int = 1280,
        height: int = 800,
        full_page: bool = False,
        wait_for: str = "networkidle",
    ):
        """
        Args:
            output_dir: Directory where PNGs will be saved (created if missing).
            width:      Viewport width in pixels.
            height:     Viewport height in pixels.
            full_page:  If True, captures the full scrollable page.
                        If False, captures only the viewport.
            wait_for:   Playwright wait_until strategy.
                        "networkidle" waits for all network activity to stop —
                        best for pages that fetch data or use a CDN (like Tailwind).
        """
        self.output_dir = Path(output_dir)
        self.width = width
        self.height = height
        self.full_page = full_page
        self.wait_for = wait_for

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def convert(self, html_path: str) -> Path:
        """Synchronous entry point. Converts one HTML file to PNG."""
        return asyncio.run(self._convert(html_path))

    def convert_many(self, html_paths: list[str]) -> list[Path]:
        """Converts multiple HTML files concurrently."""
        return asyncio.run(self._convert_many(html_paths))

    # ── internals ────────────────────────────────────────────────────────────

    async def _convert(self, html_path: str) -> Path:
        html_path = Path(html_path).resolve()
        if not html_path.exists():
            raise FileNotFoundError(f"HTML file not found: {html_path}")

        out_path = self.output_dir / (html_path.stem + ".png")
        
        print(f"Converting: {html_path} → {out_path}")

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                args=["--allow-file-access-from-files"]
            )

            page = await browser.new_page(
                viewport={"width": self.width, "height": self.height}
            )

            page.on("console", lambda msg: print(msg.text))
            page.on("pageerror", lambda err: print(err))


            await page.goto(
                html_path.as_uri(),
                wait_until="domcontentloaded"
            )

            # load JSON
            json_path = Path("./outputs/bloom_forecast.json").resolve()
            data = json.loads(json_path.read_text())

            # ensure render function exists
            await page.wait_for_function(
                "typeof startRender === 'function'",
                timeout=5000
            )

            # run render
            await page.evaluate("startRender", data)

            # wait for completion signal
            await page.wait_for_function(
                "window.renderDone === true",
                timeout=15000
            )


            await page.screenshot(
                path=str(out_path),
                full_page=self.full_page
            )


            await browser.close()

        print(f"Saved: {out_path}")
        return out_path

    async def _convert_many(self, html_paths: list[str]) -> list[Path]:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            tasks = [
                self._convert_with_browser(browser, p) for p in html_paths
            ]
            results = await asyncio.gather(*tasks)
            await browser.close()
        return list(results)

# ── Example usage ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    converter = HTMLtoPNG(
        output_dir="png_outputs",
        width=1280,
        height=800,
        full_page=True,        # captures the whole page, not just the viewport
        wait_for="networkidle",
    )

    # Single file
    converter.convert("forecast_graph.html")

    # Multiple files at once
    # converter.convert_many(["forecast_graph.html", "other_chart.html"])