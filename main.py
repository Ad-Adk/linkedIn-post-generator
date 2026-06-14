import streamlit as st
from few_shot import FewShotPosts
from post_generator import generate_post, TONE_INSTRUCTIONS, TEMPLATE_STRUCTURES

LENGTH_OPTIONS = ["Short", "Medium", "Long"]
LANGUAGE_OPTIONS = ["English", "Hinglish"]


def main():
    st.set_page_config(page_title="LinkedIn Post Generator", page_icon="✍️", layout="centered")
    st.title("✍️ LinkedIn Post Generator")
    st.caption("AI-powered posts with semantic style matching")

    fs = FewShotPosts()

    # ── Row 1: Topic | Length | Language ──────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_topic = st.selectbox("📌 Topic", options=fs.get_tags())
    with col2:
        selected_length = st.selectbox("📏 Length", options=LENGTH_OPTIONS)
    with col3:
        selected_language = st.selectbox("🌐 Language", options=LANGUAGE_OPTIONS)

    # ── Row 2: Tone | Template ─────────────────────────────────────────────────
    col4, col5 = st.columns(2)
    with col4:
        selected_tone = st.selectbox(
            "🎭 Tone",
            options=list(TONE_INSTRUCTIONS.keys()),
            help="Controls the voice and energy of the post",
        )
    with col5:
        selected_template = st.selectbox(
            "📐 Structure Template",
            options=list(TEMPLATE_STRUCTURES.keys()),
            help="Choose a structural scaffold for the post",
        )

    # ── Tone description hint ──────────────────────────────────────────────────
    with st.expander("ℹ️ About selected tone & template", expanded=False):
        st.markdown(f"**Tone — {selected_tone}:**  \n{TONE_INSTRUCTIONS[selected_tone]}")
        tmpl_desc = TEMPLATE_STRUCTURES[selected_template]
        if tmpl_desc:
            st.markdown(f"\n**Template — {selected_template}:**  \n{tmpl_desc}")
        else:
            st.markdown("**Template:** Free form — no structure enforced.")

    # ── Generate ───────────────────────────────────────────────────────────────
    if st.button("🚀 Generate Post", use_container_width=True, type="primary"):
        with st.spinner("Finding similar posts and generating..."):
            post = generate_post(
                selected_topic,
                selected_length,
                selected_language,
                selected_tone,
                selected_template,
            )

        st.markdown("---")
        st.subheader("Generated Post")
        st.write(post)

        # Copy helper
        st.code(post, language=None)
        st.caption("👆 Click the copy icon above to copy your post")


if __name__ == "__main__":
    main()
