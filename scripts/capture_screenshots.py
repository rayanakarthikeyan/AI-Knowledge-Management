import time
from pathlib import Path
from playwright.sync_api import sync_playwright

OUTPUT_DIR = Path("docs/screenshots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def capture():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="chrome",
            headless=True
        )
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        print("Navigating to http://localhost:8501...")
        page.goto("http://localhost:8501", wait_until="networkidle")
        
        # Wait for Streamlit app header
        page.wait_for_selector(".header-title", timeout=30000)
        time.sleep(2)

        # 1. Capture Dashboard initial state
        print("Capturing 01_dashboard_overview.png...")
        page.screenshot(path=str(OUTPUT_DIR / "01_dashboard_overview.png"), full_page=True)

        # 2. Click "Load Bundled Sample Documents" button
        print("Clicking Load Bundled Sample Documents...")
        sample_btn = page.get_by_role("button", name="Load Bundled Sample Documents")
        if sample_btn.is_visible():
            sample_btn.click()
            time.sleep(8)  # wait for ingestion and UI rerun

        # Wait for document list to populate
        page.wait_for_selector(".metric-number", timeout=20000)
        time.sleep(2)

        # Capture Repository Indexed state
        print("Capturing 02_indexed_repository.png...")
        page.screenshot(path=str(OUTPUT_DIR / "02_indexed_repository.png"), full_page=True)

        # 3. Submit a query to trigger multi-agent workflow
        print("Entering query in chat...")
        chat_input = page.get_by_placeholder("Enter natural language inquiry...")
        if chat_input.is_visible():
            chat_input.fill("What is the hardware and personnel budget for Project Apollo, and what are the security encryption standards?")
            chat_input.press("Enter")
            print("Waiting for agents to plan, retrieve, reason, and validate...")
            
            # Wait for assistant response and citations to appear
            page.wait_for_selector(".citation-card", timeout=45000)
            time.sleep(3)

        # Capture Multi-Agent Workflow and Citations
        print("Capturing 03_multi_agent_workflow_and_citations.png...")
        page.screenshot(path=str(OUTPUT_DIR / "03_multi_agent_workflow_and_citations.png"), full_page=True)

        browser.close()
        print("All screenshots successfully captured!")

if __name__ == "__main__":
    capture()
