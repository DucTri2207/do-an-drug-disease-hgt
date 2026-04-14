"""Dashboard trực quan tiếng Việt cho dự đoán Drug -> Disease + xác minh nhanh."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.inference import InferenceConfig, load_inference_session, predict_top_k_diseases

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
CHECKPOINTS_DIR = ROOT / "checkpoints"


@st.cache_data(show_spinner=False)
def list_result_files() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(RESULTS_DIR.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


@st.cache_data(show_spinner=False)
def list_checkpoint_files() -> list[Path]:
    if not CHECKPOINTS_DIR.exists():
        return []
    return sorted(CHECKPOINTS_DIR.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def get_session(
    checkpoint_path: str,
    model_type: str | None,
    dataset_name: str | None,
    graph_mode: str | None,
    data_root: str,
):
    return load_inference_session(
        checkpoint_path=checkpoint_path,
        dataset=dataset_name,
        model_type=model_type,
        data_root=data_root,
        config=InferenceConfig(graph_mode=graph_mode),
    )


def format_drug_option(index: int, name: str, drugbank_id: str) -> str:
    name = (name or "").strip()
    drugbank_id = (drugbank_id or "").strip()
    suffix = f" ({drugbank_id})" if drugbank_id else ""
    return f"[{index}] {name}{suffix}"


def known_associations_df(session: Any, drug_index: int) -> pd.DataFrame:
    edge = session.dataset.edges["drug_disease"]
    known_disease_indices = edge.target_index[edge.source_index == drug_index]
    if known_disease_indices.size == 0:
        return pd.DataFrame(columns=["Mã bệnh", "Tên bệnh", "Nguồn xác minh"])

    disease_labels = session.feature_bundle.lookups["disease"].labels
    rows = [
        {
            "Mã bệnh": int(i),
            "Tên bệnh": str(disease_labels[int(i)]),
            "Nguồn xác minh": "Có trong dữ liệu benchmark",
        }
        for i in sorted(set(known_disease_indices.tolist()))
    ]
    return pd.DataFrame(rows)


def prediction_df(result: Any) -> pd.DataFrame:
    rows = [
        {
            "Hạng": item.rank,
            "Mã bệnh": item.disease_index,
            "Tên bệnh": item.disease_record.get("display_name", item.disease_record.get("label", "")),
            "Xác suất": item.probability,
            "Logit": item.logit,
            "Trạng thái trong dữ liệu": "Đã biết" if item.known_association else "Mới (dự đoán)",
        }
        for item in result.predictions
    ]
    return pd.DataFrame(rows)


def verification_table(known_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df.empty and known_df.empty:
        return pd.DataFrame(columns=["Mã bệnh", "Tên bệnh", "Loại", "Xác suất", "Mức xác minh"])

    known_core = pd.DataFrame(columns=["Mã bệnh", "Tên bệnh", "Loại", "Xác suất", "Mức xác minh"])
    if not known_df.empty:
        known_core = pd.DataFrame(
            {
                "Mã bệnh": known_df["Mã bệnh"],
                "Tên bệnh": known_df["Tên bệnh"],
                "Loại": "Liên kết đã biết",
                "Xác suất": None,
                "Mức xác minh": "Có trong benchmark",
            }
        )

    pred_core = pd.DataFrame(columns=["Mã bệnh", "Tên bệnh", "Loại", "Xác suất", "Mức xác minh"])
    if not pred_df.empty:
        pred_core = pd.DataFrame(
            {
                "Mã bệnh": pred_df["Mã bệnh"],
                "Tên bệnh": pred_df["Tên bệnh"],
                "Loại": pred_df["Trạng thái trong dữ liệu"],
                "Xác suất": pred_df["Xác suất"],
                "Mức xác minh": pred_df["Trạng thái trong dữ liệu"].map(
                    lambda x: "Có trong benchmark" if x == "Đã biết" else "Cần xác minh nguồn ngoài"
                ),
            }
        )

    merged = pd.concat([known_core, pred_core], ignore_index=True)
    merged = merged.drop_duplicates(subset=["Mã bệnh", "Tên bệnh", "Loại"], keep="first")
    return merged


def render_training_overview() -> None:
    st.subheader("1) Tổng quan các lần huấn luyện")
    files = list_result_files()
    if not files:
        st.info("Chưa có file `results/*.json`. Hãy train model trước rồi quay lại dashboard.")
        return

    summaries: list[dict[str, Any]] = []
    for f in files:
        payload = load_json(str(f))
        if "test_metrics" not in payload:
            continue
        test = payload.get("test_metrics", {})
        train = payload.get("training", {})
        summaries.append(
            {
                "File": str(f.relative_to(ROOT)),
                "Dataset": payload.get("dataset", "-"),
                "Model": payload.get("model", "-"),
                "AUPR": test.get("aupr"),
                "AUC": test.get("auc"),
                "F1": test.get("f1"),
                "Best epoch": train.get("best_epoch"),
                "Epoch đã chạy": train.get("epochs_completed"),
            }
        )

    if not summaries:
        st.warning("Có file JSON nhưng không đúng định dạng summary từ `src.main`.")
        return

    table = pd.DataFrame(summaries)
    st.dataframe(table, width="stretch")

    metric = st.selectbox("Biểu đồ so sánh theo metric", ["AUPR", "AUC", "F1"], index=0)
    chart = table[["File", metric]].dropna().set_index("File")
    if not chart.empty:
        st.bar_chart(chart[metric])


def render_inference() -> None:
    st.subheader("2) Dự đoán bệnh từ tên thuốc")

    checkpoint_files = list_checkpoint_files()
    checkpoint_options = [str(p.relative_to(ROOT)) for p in checkpoint_files]
    if not checkpoint_options:
        st.error("Chưa có checkpoint trong thư mục `checkpoints/`.")
        return

    c1, c2 = st.columns([2, 1])
    with c1:
        checkpoint_rel = st.selectbox("Checkpoint mô hình", checkpoint_options, index=0)
    with c2:
        mode = st.radio("Nhận diện", ["Tự động", "Chỉnh tay"], horizontal=True)

    model_type: str | None
    dataset_name: str | None
    graph_mode: str | None

    if mode == "Chỉnh tay":
        colm1, colm2, colm3 = st.columns(3)
        with colm1:
            model_type = st.selectbox("Loại model", ["hgt", "fusion_hgt", "baseline"], index=0)
        with colm2:
            dataset_name = st.selectbox("Dataset", ["B-dataset", "C-dataset", "F-dataset"], index=1)
        with colm3:
            graph_mode = "none" if model_type == "baseline" else st.selectbox("Graph mode", ["full", "train"], index=0)
    else:
        model_type = None
        dataset_name = None
        graph_mode = None
        st.caption("Dashboard sẽ tự đọc metadata từ checkpoint.")

    colk1, colk2 = st.columns(2)
    with colk1:
        top_k = st.slider("Số bệnh gợi ý (Top-k)", 3, 30, 10, 1)
    with colk2:
        exclude_known = st.checkbox("Chỉ hiện bệnh mới (ẩn liên kết đã biết)", value=True)

    checkpoint_path = ROOT / checkpoint_rel
    try:
        with st.spinner("Đang nạp model + dữ liệu..."):
            session = get_session(
                checkpoint_path=str(checkpoint_path),
                model_type=model_type,
                dataset_name=dataset_name,
                graph_mode=graph_mode,
                data_root=str(ROOT / "data"),
            )
    except Exception as exc:
        st.exception(exc)
        return

    drug_labels = session.feature_bundle.lookups["drug"].labels
    drugbank_ids = session.feature_bundle.lookups["drug"].metadata.get("drugbank_id", [""] * session.dataset.node_counts["drug"])
    all_drugs = [{"index": i, "name": str(drug_labels[i]), "drugbank_id": str(drugbank_ids[i])} for i in range(session.dataset.node_counts["drug"])]

    query = st.text_input("Tìm thuốc theo tên / DrugBank ID", "")
    if query.strip():
        q = query.strip().casefold()
        all_drugs = [d for d in all_drugs if q in d["name"].casefold() or q in d["drugbank_id"].casefold()]

    if not all_drugs:
        st.warning("Không tìm thấy thuốc phù hợp.")
        return

    option_map = {format_drug_option(d["index"], d["name"], d["drugbank_id"]): d for d in all_drugs}
    selected_label = st.selectbox("Chọn thuốc", list(option_map.keys()), index=0)
    selected_drug = option_map[selected_label]
    drug_index = int(selected_drug["index"])

    st.markdown("### Thông tin thuốc")
    m1, m2, m3 = st.columns(3)
    m1.metric("Mã nội bộ", drug_index)
    m2.metric("Tên thuốc", selected_drug["name"] or "(trống)")
    m3.metric("DrugBank ID", selected_drug["drugbank_id"] or "(không có)")
    if selected_drug["drugbank_id"]:
        st.link_button("Mở trang DrugBank", f"https://go.drugbank.com/drugs/{selected_drug['drugbank_id']}")

    st.markdown("### Liên kết đã biết trong dữ liệu")
    known_df = known_associations_df(session, drug_index)
    if known_df.empty:
        st.info("Thuốc này chưa có liên kết bệnh đã biết trong dataset hiện tại.")
    else:
        st.dataframe(known_df, width="stretch")

    if st.button("Dự đoán bệnh tiềm năng", type="primary"):
        try:
            result = predict_top_k_diseases(
                session,
                drug_query=drug_index,
                top_k=top_k,
                exclude_known_associations=exclude_known,
            )
        except Exception as exc:
            st.exception(exc)
            return

        pred_df = prediction_df(result)
        if pred_df.empty:
            st.warning("Không có bệnh nào sau khi lọc.")
            return

        cstat1, cstat2 = st.columns(2)
        cstat1.metric("Số ứng viên đã xét", result.num_candidates_considered)
        cstat2.metric("Số liên kết đã biết bị loại", result.num_known_filtered)

        st.markdown("### Kết quả dự đoán")
        st.dataframe(pred_df, width="stretch")
        st.bar_chart(pred_df.set_index("Tên bệnh")[["Xác suất"]])

        st.markdown("### Bảng xác minh tổng hợp")
        verify_df = verification_table(known_df, pred_df)
        st.dataframe(verify_df, width="stretch")
        st.download_button(
            "Tải CSV bảng xác minh",
            data=verify_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"verify_drug_{drug_index}.csv",
            mime="text/csv",
        )

        st.warning(
            "Lưu ý học thuật: 'Mới (dự đoán)' là liên kết do mô hình gợi ý, cần xác minh thêm bằng nguồn ngoài/literature trước khi kết luận y sinh."
        )


def main() -> None:
    st.set_page_config(page_title="Dashboard Dự đoán Thuốc - Bệnh", layout="wide")
    st.title("Dashboard Dự đoán Thuốc - Bệnh")
    st.caption("Chọn thuốc theo tên thật, xem bệnh đã biết và bệnh tiềm năng một cách trực quan.")

    tab1, tab2 = st.tabs(["Tổng quan huấn luyện", "Dự đoán theo tên thuốc"])
    with tab1:
        render_training_overview()
    with tab2:
        render_inference()


if __name__ == "__main__":
    main()
