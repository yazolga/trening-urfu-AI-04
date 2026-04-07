import io
import streamlit as st
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForVision2Seq


MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"


# Функция загрузки модели и токенизатора
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        # Пока не используем AutoProcessor вообще
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_NAME,
            torch_dtype=dtype,
            _attn_implementation="eager",
        ).to(device)

        model.eval()
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Ошибка при загрузке модели: {e}")
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


# Функция распознавания через tokenizer (без AutoProcessor и model.processor)
def transcribe_image(tokenizer, model, device, image):
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

    # Попробуем apply_chat_template; если его нет — упростим prompt
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except AttributeError:
        # Если apply_chat_template не поддерживается, просто склеим текст
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            tokenize=False,
        )
        if isinstance(prompt, str):
            pass
        else:
            # Простой резервный вариант
            prompt = "User:\n<image>\nTranscribe the text exactly as it appears in the image."

    # Обычный токенизинг текста
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Попытка использовать возможный .processor модели (если он есть)
    try:
        # Если у модели есть .processor:
        pixel_values = model.processor(images=[image], return_tensors="pt")["pixel_values"].to(device)
        inputs["pixel_values"] = pixel_values
    except AttributeError:
        # Если .processor нет — продолжаем без pixel_values
        st.info("Модель или версия transformers не поддерживает .processor. Попробуем без него.")
        pass

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[:, prompt_length:]

    generated_text = tokenizer.batch_decode(
        new_tokens,
        skip_special_tokens=True
    )[0].strip()

    return generated_text


# Основная часть приложения
st.set_page_config(page_title="Распознать английский текст с изображения!")
st.title("🌟 Распознать английский текст с изображения!")
st.write("Загрузите изображение и нажмите кнопку распознавания.")

tokenizer, model, device = load_model()
if tokenizer is None or model is None:
    st.warning("Модель не загружена, проверьте зависимость `transformers` и интернет‑соединение.")
    st.stop()

img = load_image()

if st.button("Распознать изображение", type="primary"):
    if img is None:
        st.warning("Сначала загрузите изображение.")
    else:
        with st.spinner("Распознавание текста..."):
            try:
                result = transcribe_image(tokenizer, model, device, img)
                st.success("✅ Распознавание завершено!")
                st.markdown(f"**Распознанный текст:** {result}")
            except Exception as e:
                st.error(f"Ошибка при распознавании: {e}")
