import io
import streamlit as st
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"


# Функция загрузки модели
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_NAME,
            torch_dtype=dtype,
            _attn_implementation="eager",
        ).to(device)
        model.eval()
        return processor, model, device
    except Exception as e:
        st.error(f"Ошибка при загрузке модели: {e}")
        st.stop()


# Функция загрузки изображения через Streamlit
def load_image():
    uploaded_file = st.file_uploader(
        "Выберите изображение для распознавания",
        type=["png", "jpg", "jpeg", "webp"]
    )

    if uploaded_file is not None:
        # Ограничение размера файла (5 МБ)
        if uploaded_file.size > 5 * 1024 * 1024:
            st.warning("Размер изображения больше 5 МБ. Загрузите меньшее изображение.")
            return None

        image_data = uploaded_file.getvalue()
        st.image(image_data, caption="Загруженное изображение", width=400)

        try:
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            image.thumbnail((800, 800))  # уменьшаем размер изображения
            return image
        except Exception as e:
            st.error(f"Ошибка при загрузке изображения: {e}")
            return None

    return None


# Функция распознавания
def transcribe_image(processor, model, device, image):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        "Transcribe the text exactly as it appears in the image. "
                        "Do not paraphrase. "
                        "Do not add explanations. "
                        "Output only the recognized text."
                    )
                },
            ],
        }
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[:, prompt_length:]

    generated_text = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True
    )[0].strip()

    return generated_trext


# Основная часть приложения
st.set_page_config(page_title="Распознать английский текст с изображения!")
st.title("🌟 Распознать английский текст с изображения!")
st.write("Загрузите изображение и нажмите кнопку распознавания.")

# Проверка, что модель загружена корректно
processor, model, device = load_model()
if processor is None or model is None:
    st.warning("Модель не загружена, проверьте зависимости и интернет‑соединение.")
    st.stop()

img = load_image()

# Распознавание и вывод результата
if st.button("Распознать изображение", type="primary"):
    if img is None:
        st.warning("Сначала загрузите изображение.")
    else:
        with st.spinner("Распознавание текста..."):
            try:
                result = transcribe_image(processor, model, device, img)
                st.success("✅ Распознавание завершено!")
                st.markdown(f"**Распознанный текст:** {result}")
            except Exception as e:
                st.error(f"Ошибка при распознавании: {e}")
