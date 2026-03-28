"""Chat page — test local model generation interactively."""

import streamlit as st


def render() -> None:
    st.header("Chat")

    # Model loading state
    if "model" not in st.session_state:
        st.session_state.model = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # --- Model controls ---
    with st.sidebar:
        st.subheader("Model Settings")
        system_prompt = st.text_area(
            "System prompt",
            value="You are a helpful AI research assistant skilled in code generation and paper analysis.",
            height=80,
        )
        max_tokens = st.slider("Max tokens", 64, 4096, 1024, step=64)
        temperature = st.slider("Temperature", 0.0, 1.5, 0.1, step=0.05)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Load Model", use_container_width=True):
                _load_model()
        with col2:
            if st.button("Unload", use_container_width=True):
                _unload_model()

        if st.session_state.model is not None and st.session_state.model.is_loaded:
            usage = st.session_state.model.vram_usage()
            st.success(f"Model loaded — {usage['allocated_gb']} GB VRAM")
        else:
            st.warning("Model not loaded")

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # --- Chat display ---
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- Chat input ---
    if prompt := st.chat_input("Ask anything..."):
        if st.session_state.model is None or not st.session_state.model.is_loaded:
            st.error("Load the model first (sidebar).")
            return

        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response with streaming
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""

            messages = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.chat_history
            ]

            for token in st.session_state.model.chat_stream(
                messages=messages,
                system=system_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
            ):
                full_response += token
                placeholder.markdown(full_response + "\u2588")

            placeholder.markdown(full_response)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": full_response}
        )


def _load_model() -> None:
    """Load the local model with a progress indicator."""
    with st.spinner("Loading Qwen2.5-Coder-7B (NF4)..."):
        from papermind.infrastructure.local_model import LocalModel
        model = LocalModel()
        model.load()
        st.session_state.model = model


def _unload_model() -> None:
    """Unload model and free VRAM."""
    if st.session_state.model is not None:
        st.session_state.model.unload()
        st.session_state.model = None
    st.rerun()
