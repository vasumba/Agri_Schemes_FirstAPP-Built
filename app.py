
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8001"

st.set_page_config(
    page_title="Agri Schemes RAG",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 Agricultural Schemes AI Assistant")
st.write("Upload agricultural scheme PDFs and ask questions about them.")

left, right = st.columns(2, gap="large")

with left:
    st.subheader("📄 Upload PDFs")

    uploaded_files = st.file_uploader(
        "Choose agricultural scheme PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Process PDFs", type="primary"):
        if not uploaded_files:
            st.warning("Please select at least one PDF.")
        else:
            files = [
                (
                    "files",
                    (file.name, file.getvalue(), "application/pdf")
                )
                for file in uploaded_files
            ]

            try:
                with st.spinner("Processing PDFs..."):
                    response = requests.post(
                        f"{API_URL}/upload",
                        files=files,
                        timeout=300
                    )
                    response.raise_for_status()

                result = response.json()
                st.success(result.get("message", "Upload completed"))
                st.write("New files:", result.get("new_files", 0))
                st.write("Chunks added:", result.get("chunks_added", 0))

            except requests.RequestException as error:
                st.error(f"Upload failed: {error}")

with right:
    st.subheader("💬 Ask Questions")

    question = st.text_area(
        "Enter your question about agricultural schemes"
    )

    if st.button("Get Answer", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Finding the answer..."):
                    response = requests.post(
                        f"{API_URL}/ask",
                        json={"question": question},
                        timeout=120
                    )
                    response.raise_for_status()

                result = response.json()

                st.subheader("Answer")
                st.write(result.get("answer", "No answer returned."))

                st.subheader("Sources")
                sources = result.get("sources", [])

                if sources:
                    for source in sources:
                        st.write(
                            f"📄 {source['filename']} — "
                            f"Page {source['page']}"
                        )
                else:
                    st.info("No sources returned.")

            except requests.RequestException as error:
                st.error(f"Question failed: {error}")
