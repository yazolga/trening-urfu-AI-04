import io
import streamlit as st
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq


MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"


# Функция загрузки модели и процессора
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
        except Exception as e:
            st.error(f"Ошибка при загрузке изображения: {e}")
            return None

    return None


# Функция распознавания — через processor, а не model.processor
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

    # Применяем чат‑шаблон процессора
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    # Объединяем текст и изображение в один вход через processor
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

    return generated_text


# Основная часть приложения
st.set_page_config(page_title="Распознать английский текст с изображения!")
st.title("🌟 Распознать английский текст с изображения!")
st.write("Загрузите изображение и нажмите кнопку распознавания.")

# Загружаем processor и model одновременно
processor, model, device = load_model()
if processor is None or model is None:
    st.warning("Модель или процессор не загружены, проверьте зависимость `transformers` и интернет‑соединение.")
    st.stop()

img = load_image()

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
