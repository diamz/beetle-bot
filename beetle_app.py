import streamlit as st
import pandas as pd
from beetlebot_module import run_crawler

st.set_page_config(page_title="BeetleBot", layout="wide")
st.title("🐞 BeetleBot")

# --- User input (only URL)
url = st.text_input("Enter URL:", "https://example.com/")

# --- Hard-coded max pages
MAX_PAGES = 500

if st.button("Run Crawler"):
    progress_bar = st.progress(0)
    status_text = st.empty()

    # progress callback from crawler
    def progress_callback(count, rec):
        pct = int(count / MAX_PAGES * 100)
        progress_bar.progress(min(pct, 100))
        status_text.text(f"📄 Scanning page {count}: {rec['url']}")

    with st.spinner("Crawling..."):
        results = run_crawler(
            url,
            max_pages=MAX_PAGES,
            out_dir=None,
            progress_callback=progress_callback
        )

    st.success(f"✅ Done! Crawled {len(results)} pages (limit {MAX_PAGES}).")

    if results:
        df = pd.DataFrame(results)
        light_df = df[["url", "status", "title", "meta_description", "depth"]]
        st.dataframe(light_df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV",
            csv,
            "crawler_results.csv",
            "text/csv"
        )
    else:
        st.warning("No pages were crawled.")



