"""Chat page — RAG-augmented generation with local model.

Two modes:
  1. RAG Chat: Question → retrieve → rerank → assemble context → LLM generation
  2. Direct Chat: Standalone LLM conversation (no retrieval)
"""

import asyncio
import re

import streamlit as st


def _render_latex(text: str) -> str:
    r"""Convert LLM LaTeX output to Streamlit-compatible format.

    LLMs output:  \( x^2 \)  and  \[ E = mc^2 \]  and  ( \mathcal{F} )
    Streamlit needs:  $x^2$  and  $$E = mc^2$$
    """
    # Display math: \[ ... \] → $$ ... $$
    text = re.sub(r'\\\[\s*', '\n$$\n', text)
    text = re.sub(r'\s*\\\]', '\n$$\n', text)
    # Inline math: \( ... \) → $ ... $
    text = re.sub(r'\\\(\s*', '$', text)
    text = re.sub(r'\s*\\\)', '$', text)
    return text


def render() -> None:
    st.header("Chat")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "rag_sources" not in st.session_state:
        st.session_state.rag_sources = {}

    # --- Sidebar controls ---
    with st.sidebar:
        st.subheader("Chat Settings")

        mode = st.radio(
            "Mode",
            ["RAG (paper-grounded)", "Direct (no retrieval)"],
            index=0,
            horizontal=True,
        )
        use_rag = "RAG" in mode

        if use_rag:
            st.caption("Retrieves relevant paper chunks, reranks, and generates a grounded answer.")
            with st.expander("RAG Settings", expanded=False):
                n_retrieve = st.slider("Initial retrieval count", 10, 50, 30, step=5)
                rerank_top_k = st.slider("Reranked results to keep", 3, 20, 10)
                token_budget = st.slider("Context token budget", 1024, 8192, 4096, step=512)
                use_kg = st.checkbox("Include knowledge graph", value=True)
                use_rerank = st.checkbox("Enable cross-encoder reranking", value=True)

                # Paper filter — load from persistent store
                from papermind.ui.shared import get_paper_store
                paper_store = get_paper_store()
                stored_papers = paper_store.get_paper_map()
                paper_options = {"All papers": None}
                paper_options.update({
                    p.title[:50] or f"Paper {pid[:8]}": pid
                    for pid, p in stored_papers.items()
                })
                paper_filter = st.selectbox(
                    "Filter to paper",
                    options=list(paper_options.keys()),
                )
                paper_id = paper_options[paper_filter]
        else:
            st.caption("Direct conversation with the LLM — no paper context.")

        system_prompt = st.text_area(
            "System prompt",
            value=(
                "You are a helpful AI research assistant skilled in "
                "code generation and paper analysis."
            ),
            height=80,
        )
        max_tokens = st.slider("Max tokens", 64, 4096, 1024, step=64)
        temperature = st.slider("Temperature", 0.0, 1.5, 0.1, step=0.05)

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.rag_sources = {}
            st.rerun()

    # --- Chat display ---
    for i, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            content = _render_latex(msg["content"]) if msg["role"] == "assistant" else msg["content"]
            st.markdown(content)

            # Show sources for RAG responses
            if msg["role"] == "assistant" and i in st.session_state.rag_sources:
                sources = st.session_state.rag_sources[i]
                if sources:
                    with st.expander(f"Sources ({len(sources)})"):
                        for src in sources:
                            score = src.get("score", 0)
                            section = src.get("section_title", "")
                            text_preview = src.get("text", "")[:200]
                            st.markdown(
                                f"**{section}** (score: {score:.3f})\n\n"
                                f"> {text_preview}..."
                            )

    # --- Chat input ---
    if prompt := st.chat_input("Ask about your papers..." if use_rag else "Ask anything..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        msg_index = len(st.session_state.chat_history)

        with st.chat_message("assistant"):
            if use_rag:
                full_response, sources = _rag_generate(
                    prompt,
                    system_prompt=system_prompt,
                    n_retrieve=n_retrieve,
                    rerank_top_k=rerank_top_k,
                    token_budget=token_budget,
                    use_kg=use_kg,
                    use_rerank=use_rerank,
                    paper_id=paper_id,
                )
                st.session_state.rag_sources[msg_index] = sources
            else:
                full_response = _direct_generate(
                    prompt,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

        st.session_state.chat_history.append(
            {"role": "assistant", "content": full_response}
        )
        st.rerun()


def _rag_generate(
    question: str,
    system_prompt: str = "",
    n_retrieve: int = 30,
    rerank_top_k: int = 10,
    token_budget: int = 4096,
    use_kg: bool = True,
    use_rerank: bool = True,
    paper_id: str | None = None,
) -> tuple[str, list[dict]]:
    """Run the RAG pipeline synchronously and return (answer, sources).

    Calls retrieval, reranking, and context assembly directly (all sync),
    then tries LLM generation via Ollama/local model.
    """
    from papermind.rag.retriever import hybrid_retrieve
    from papermind.rag.context import assemble_context
    from papermind.rag.pipeline import DEFAULT_SYSTEM_PROMPT

    status = st.status("Running RAG pipeline...", expanded=True)

    # Stage 1: Hybrid retrieval
    status.write(f"Retrieving top {n_retrieve} results...")
    retrieved = hybrid_retrieve(
        query=question,
        n_results=n_retrieve,
        paper_id=paper_id,
        use_kg=use_kg,
    )

    if not retrieved:
        status.update(label="No results found", state="error", expanded=False)
        answer = (
            "I couldn't find any relevant information in the ingested papers. "
            "Try ingesting more papers or rephrasing your question."
        )
        st.markdown(_render_latex(answer))
        return answer, []

    status.write(f"Retrieved {len(retrieved)} results")

    # Stage 2: Reranking
    if use_rerank and len(retrieved) > 1:
        status.write(f"Reranking {len(retrieved)} results with cross-encoder...")
        from papermind.rag.reranker import rerank
        retrieved = rerank(
            query=question,
            results=retrieved,
            top_k=rerank_top_k,
        )
        status.write(f"Kept top {len(retrieved)} after reranking")

    # Stage 3: Context assembly
    status.write("Assembling context...")
    context_str, context_results = assemble_context(
        results=retrieved,
        token_budget=token_budget,
    )

    status.write(
        f"{len(context_results)} sources in context "
        f"({len(context_str)} chars)"
    )

    # Free GPU memory from reranker before loading LLM
    if use_rerank:
        from papermind.rag.reranker import get_reranker
        get_reranker().unload()

    # Stage 4: LLM generation
    system = system_prompt or DEFAULT_SYSTEM_PROMPT
    prompt = (
        f"Context from research papers:\n\n"
        f"{context_str}\n\n"
        f"---\n\n"
        f"Question: {question}\n\n"
        f"Answer based on the context above. Be specific and cite the sources."
    )

    status.write("Generating answer...")

    answer = ""
    try:
        from papermind.infrastructure.llm_client import LLMClient
        client = LLMClient()
        answer = asyncio.run(client.generate(prompt, system=system))
    except RuntimeError:
        # Already in an async event loop (Streamlit) — use nest_asyncio or thread
        import concurrent.futures
        from papermind.infrastructure.llm_client import LLMClient
        client = LLMClient()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            answer = pool.submit(
                asyncio.run, client.generate(prompt, system=system)
            ).result(timeout=120)
    except Exception as e:
        answer = (
            f"Found {len(context_results)} relevant sources but LLM generation "
            f"failed ({e}). Here are the top sources:\n\n"
            + "\n\n".join(
                f"**{r.section_title}** (score: {r.score:.2f}): {r.text[:200]}..."
                for r in context_results[:3]
            )
        )

    status.update(label="RAG pipeline complete", state="complete", expanded=False)

    st.markdown(_render_latex(answer))

    # Format sources
    sources = [
        {
            "text": s.text,
            "score": s.score,
            "section_title": s.section_title,
            "paper_id": s.paper_id,
        }
        for s in context_results
    ]

    if context_results:
        with st.expander(f"Sources ({len(context_results)})"):
            for s in context_results:
                st.markdown(
                    f"**{s.section_title}** (score: {s.score:.3f})\n\n"
                    f"> {s.text[:200]}..."
                )

    return answer, sources


def _direct_generate(
    prompt: str,
    system_prompt: str = "",
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> str:
    """Direct LLM generation without RAG context."""
    # Check if local model is loaded
    if "model" in st.session_state and st.session_state.model is not None:
        model = st.session_state.model
        if model.is_loaded:
            placeholder = st.empty()
            full_response = ""
            messages = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.chat_history
            ]
            for token in model.chat_stream(
                messages=messages,
                system=system_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
            ):
                full_response += token
                placeholder.markdown(_render_latex(full_response) + "\u2588")
            placeholder.markdown(_render_latex(full_response))
            return full_response

    # Try Ollama backend
    from papermind.infrastructure.llm_client import LLMClient
    client = LLMClient()

    try:
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_history
        ]
        result = asyncio.run(client.chat(messages, system=system_prompt))
        st.markdown(_render_latex(result))
        return result
    except Exception as e:
        error_msg = (
            f"LLM not available. Either:\n"
            f"- Load the local model (sidebar in Direct mode)\n"
            f"- Start Ollama: `ollama serve`\n\n"
            f"Error: {e}"
        )
        st.error(error_msg)
        return error_msg
