import whisper
import subprocess
from tempfile import TemporaryDirectory
import os
import torch
from rich.console import Console
from rich.progress import Progress
from openai import OpenAI
import logging
from dotenv import load_dotenv
from rich.markdown import Markdown

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YouTube_Transcripts:

    def __init__(
            self,
            whisper_model: str = "base",  # Whisper model size (e.g., "base", "small", "medium", "large")
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Whisper is running on: {device}")
        self.whisper_model = whisper.load_model(whisper_model, device=device)

        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        if not deepseek_api_key:
            raise ValueError("DeepSeek API key not found in .env file. Please add DEEPSEEK_API_KEY to your .env file.")

        self.deepseek_client = OpenAI(
            api_key=deepseek_api_key,
            base_url="https://api.deepseek.com/v1",
        )

    def download_audio(self, url: str, download_dir: str) -> str:
        output_path = os.path.join(download_dir, "audio.%(ext)s")
        command = [
            "yt-dlp", "-x", "--audio-format", "mp3",
            "-o", output_path, url
        ]
        try:
            subprocess.run(command, check=True)
            return os.path.join(download_dir, "audio.mp3")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download audio: {e}")
            raise ValueError(f"Invalid YouTube URL or download failed: {url}")

    def transcribe_audio(self, audio_path: str) -> str:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        result = self.whisper_model.transcribe(audio_path)
        logger.info(f"Transcription complete. Length: {len(result['text'])} characters")
        return result["text"]

    def explain_transcript(self, text: str) -> str:
        try:
            # Use DeepSeek to explain the transcript
            response = self.deepseek_client.chat.completions.create(
                model="deepseek-chat",  # Use the deepseek-chat model
                messages=[
                    {"role": "system",
                     "content": "You are a technical explainer tasked with converting spoken transcripts into precise, structured, and content-rich explanations. Your goal is to explain—not summarize—by reconstructing the full logic, detail, and flow of ideas for knowledge retrieval purposes. Do not skip or condense concepts. Focus on clarity, structure, and fidelity to the original content."},
                    {"role": "user", "content": f"Explain the following transcript in detail:\n{text}"},
                ],
                temperature= 0.7,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}")
            raise

    def process(self, url: str) -> str:
        try:
            with Progress() as progress:
                task1 = progress.add_task("Downloading audio...", total=1)
                task2 = progress.add_task("Transcribing audio...", total=1)
                task3 = progress.add_task("Generating explanation...", total=1)

                with TemporaryDirectory() as temp_dir:
                    progress.update(task1, advance=1)
                    audio_path = self.download_audio(url, temp_dir)

                    progress.update(task2, advance=1)
                    transcript = self.transcribe_audio(audio_path)

                    progress.update(task3, advance=1)
                    explanation = self.explain_transcript(transcript).strip()

                    return explanation
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise


# === Example Usage ===
if __name__ == "__main__":
    processor = YouTube_Transcripts(
        whisper_model="base",
    )
    url = "https://youtu.be/Xdv83MFJd7U?si=mfvNVohyxCr56sXr"  # Replace with your YouTube link
    try:
        explanation = processor.process(url)
        console = Console()
        md = Markdown(explanation)
        console.print("\n\n---  Explanation Start ---\n")
        console.print(md)
        console.print("\n---  Explanation End ---")

    except Exception as e:
        print("Error:", e)