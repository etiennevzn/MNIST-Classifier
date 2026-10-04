import streamlit as st

from pixel_canvas import pixel_canvas
from predictor import MODEL_PATH, Predictor

st.set_page_config(page_title="MNIST Digit Classifier", layout="centered")

STYLE = """
<style>
.block-container {
    max-width: 860px;
    padding-top: 3rem;
}

.prediction {
    border: 1px solid rgba(128, 128, 128, 0.35);
    border-radius: 8px;
    padding: 1.25rem 1rem;
    text-align: center;
    margin-bottom: 1.5rem;
}

.prediction-label {
    font-size: 0.9rem;
    opacity: 0.7;
}

.prediction-digit {
    font-size: 4.5rem;
    font-weight: 700;
    line-height: 1.1;
}

.prediction-confidence {
    font-size: 0.9rem;
    opacity: 0.7;
}

.prob-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.45rem;
}

.prob-label {
    width: 1rem;
    font-weight: 600;
    text-align: right;
}

.bar-track {
    flex: 1;
    height: 0.6rem;
    border-radius: 0.3rem;
    background: rgba(128, 128, 128, 0.2);
    overflow: hidden;
}

.bar-fill {
    height: 100%;
    border-radius: 0.3rem;
    background: rgba(128, 128, 128, 0.6);
}

.bar-fill.highlight {
    background: #1f6feb;
}

.prob-value {
    width: 4.5rem;
    text-align: right;
    font-variant-numeric: tabular-nums;
}

.empty-state {
    border: 1px dashed rgba(128, 128, 128, 0.4);
    border-radius: 8px;
    padding: 2rem 1rem;
    text-align: center;
    opacity: 0.7;
}
</style>
"""


@st.cache_resource
def load_predictor():
    return Predictor()


def reset_canvas():
    st.session_state.canvas_version += 1
    st.session_state.result = None


def render_prediction(probabilities):
    predicted = int(probabilities.argmax())
    confidence = probabilities[predicted] * 100

    st.markdown(
        f"""
        <div class="prediction">
            <div class="prediction-label">Predicted digit</div>
            <div class="prediction-digit">{predicted}</div>
            <div class="prediction-confidence">Confidence {confidence:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rows = []
    for digit, probability in enumerate(probabilities):
        fill_class = "bar-fill highlight" if digit == predicted else "bar-fill"
        rows.append(
            f'<div class="prob-row">'
            f'<span class="prob-label">{digit}</span>'
            f'<div class="bar-track">'
            f'<div class="{fill_class}" style="width:{probability * 100:.2f}%"></div>'
            f"</div>"
            f'<span class="prob-value">{probability * 100:.2f}%</span>'
            f"</div>"
        )

    st.markdown("".join(rows), unsafe_allow_html=True)


if "canvas_version" not in st.session_state:
    st.session_state.canvas_version = 0

if "result" not in st.session_state:
    st.session_state.result = None

st.markdown(STYLE, unsafe_allow_html=True)

st.title("MNIST digit classifier")
st.caption("Draw a digit on the grid, then run the model to see its prediction.")

if not MODEL_PATH.exists():
    st.error(
        f"Model file not found at {MODEL_PATH}. "
        "Run main.py once to train the model and generate it."
    )
    st.stop()

predictor = load_predictor()

left, right = st.columns(2, gap="large")

with left:
    st.subheader("Input")
    pixels = pixel_canvas(key=f"canvas_{st.session_state.canvas_version}")

    reset_column, predict_column = st.columns(2)
    reset_column.button("Reset", on_click=reset_canvas, use_container_width=True)
    predict_clicked = predict_column.button(
        "Predict", type="primary", use_container_width=True
    )

if predict_clicked:
    if pixels is None or max(pixels) == 0:
        st.session_state.result = None
        with left:
            st.warning("Draw a digit before running the prediction.")
    else:
        st.session_state.result = {
            "pixels": pixels,
            "probabilities": predictor.predict(pixels),
        }

with right:
    st.subheader("Prediction")
    result = st.session_state.result

    if result is not None and result["pixels"] == pixels:
        render_prediction(result["probabilities"])
    else:
        st.markdown(
            '<div class="empty-state">Draw a digit and click Predict.</div>',
            unsafe_allow_html=True,
        )
