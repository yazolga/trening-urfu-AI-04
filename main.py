import io
import streamlit as st
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoTokenizer, PreTrainedTokenizer, AutoProcessor


MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"


# Функция загрузки модели и процессора
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        # Попробуем AutoProcessor — если есть, пользуем его
        try:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(MODEL_NAME)
        except ImportError:
            st.error("AutoProcessor не найден в данной версии transformers. Попробуем AutoTokenizer.")
            processor = None

        # Если AutoProcessor не работает, используем AutoTokenizer
        if processor is None:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            processor = AutoProcessor(
                image_processor=tokenizer.image_processor if hasattr(tokenizer, "image_processor") else None,
                tokenizer=tokenizer,
            )

        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_NAME,
            torch_dtype=dtype,
            _attn_implementation="eager",
        ).to(device)

        model.eval()
        return processor, model, device
    except Exception as e:
        st.error(f"Ошибка при загрузке модели/процессора: {e}")
        st.stop()


# Функция загрузки изображения
def load_image():
    uploaded_file = st.file_uploader(
        "Выберите изображение для распознавания",
        type=["png", "jpg", "jpeg", "webp"]
    )

    if uploaded_file is not None:
        if uploaded_file.size > 5 * 1024 * 1024:
            st.warning("Размер изображения больше 5 МБ. Загрузите меньшее изображение.")
            return None

        image_data = uploaded_file.getvalue()
        st.image(image_data, caption="Загруженное изображение", width=400)

        try:
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            image.thumbnail((800, 800))
            return image
        except Excep
