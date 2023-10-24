import whisper
model = whisper.load_model("base")
import gradio as gr
import time
def transcribe(audio):

    time.sleep(3)
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    options = whisper.DecodingOptions(fp16 = False)
    result = whisper.decode(model, mel, options)
    return result.text


gr.Interface(
    title = 'Speech To Text',
    fn=transcribe,
    inputs=[
        gr.inputs.Audio(source="microphone", type="filepath")
    ],
    outputs=[
        "textbox"
    ],
    live=True).launch()
