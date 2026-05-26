# Updated render_gallery function for responsive grid layout
# Replace existing render_gallery with this version

def render_gallery(results: List[Dict], preview_count: int, columns_per_row: int, preview_width: int, layout_mode: str):
    visible_results = results[:preview_count]

    for start in range(0, len(visible_results), columns_per_row):
        row_items = visible_results[start:start + columns_per_row]
        cols = st.columns(columns_per_row)

        for col, item in zip(cols, row_items):
            with col:
                st.markdown(
                    f"""
                    <div style='
                        border:1px solid #444;
                        border-radius:10px;
                        padding:10px;
                        margin-bottom:12px;
                        background-color:#1e1e1e;
                    '>
                    <div style='font-size:13px;font-weight:bold;margin-bottom:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>
                        {item['name']}
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if layout_mode == "결과만 갤러리":
                    st.image(
                        resize_for_preview(item["adjusted_rgb"], preview_width),
                        width=preview_width,
                    )

                elif layout_mode == "원본/결과 2열 비교":
                    sub_left, sub_right = st.columns(2)

                    with sub_left:
                        st.caption("Original")
                        st.image(
                            resize_for_preview(item["original_rgb"], preview_width // 2),
                            width=preview_width // 2,
                        )

                    with sub_right:
                        st.caption("Adjusted")
                        st.image(
                            resize_for_preview(item["adjusted_rgb"], preview_width // 2),
                            width=preview_width // 2,
                        )

                else:
                    sub_left, sub_center, sub_right = st.columns(3)

                    with sub_left:
                        st.caption("Original")
                        st.image(
                            resize_for_preview(item["original_rgb"], preview_width // 3),
                            width=preview_width // 3,
                        )

                    with sub_center:
                        st.caption("Gray")
                        st.image(
                            resize_for_preview(
                                rgb_to_gray(item["original_rgb"]),
                                preview_width // 3,
                            ),
                            width=preview_width // 3,
                            clamp=True,
                        )

                    with sub_right:
                        st.caption("Adjusted")
                        st.image(
                            resize_for_preview(item["adjusted_rgb"], preview_width // 3),
                            width=preview_width // 3,
                        )
