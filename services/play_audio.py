import asyncio
from pathlib import Path
from typing import Optional
from openai import AsyncOpenAI, OpenAI
from openai.helpers import LocalAudioPlayer
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class AudioService:
    def __init__(self):
        """Initialize the AudioService with OpenAI client."""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.default_model = "gpt-4o-mini-tts"
        self.default_voice = "coral"
        self.default_format = "mp3"
        self.output_dir = Path(__file__).parent.parent / "data"  # Save files in backend/data

    async def generate_audio(
        self,
        text: str,
        voice: str = "coral",
        model: str = "gpt-4o-mini-tts",
        response_format: str = "mp3",
        instructions: Optional[str] = None,
        output_file: Optional[str] = None
    ) -> dict:
        """
        Generate audio from text and save to a file or return the file path.
        
        Args:
            text (str): The input text to convert to speech.
            voice (str): The voice to use (e.g., 'coral', 'alloy'). Defaults to 'coral'.
            model (str): The TTS model to use. Defaults to 'gpt-4o-mini-tts'.
            response_format (str): The audio format (e.g., 'mp3', 'wav'). Defaults to 'mp3'.
            instructions (Optional[str]): Additional instructions for tone, etc.
            output_file (Optional[str]): Name of the output file. If None, a default name is used.
        
        Returns:
            dict: Contains status, file_path (if saved), and error message (if any).
        """
        try:
            # Ensure output directory exists
            self.output_dir.mkdir(exist_ok=True)
            
            # Set default output file path if not provided
            if output_file is None:
                output_file = f"speech_{int(asyncio.get_event_loop().time())}.{response_format}"
            speech_file_path = self.output_dir / output_file

            # Generate audio using OpenAI API
            with self.client.audio.speech.with_streaming_response.create(
                model=model,
                voice=voice,
                input=text,
                instructions=instructions,
                response_format=response_format
            ) as response:
                response.stream_to_file(speech_file_path)

            return {
                "status": "success",
                "file_path": str(speech_file_path),
                "message": f"Audio generated and saved to {speech_file_path}"
            }

        except Exception as e:
            return {
                "status": "error",
                "file_path": None,
                "message": f"Failed to generate audio: {str(e)}"
            }

    async def stream_audio(
        self,
        text: str,
        voice: str = "coral",
        model: str = "gpt-4o-mini-tts",
        response_format: str = "mp3",  # Changed default to mp3
        instructions: Optional[str] = None
    ) -> dict:
        """
        Get audio stream data for client-side playback.
        
        Args:
            text (str): The input text to convert to speech.
            voice (str): The voice to use (e.g., 'coral', 'alloy'). Defaults to 'coral'.
            model (str): The TTS model to use. Defaults to 'gpt-4o-mini-tts'.
            response_format (str): The audio format for streaming (e.g., 'mp3', 'wav'). Defaults to 'mp3'.
            instructions (Optional[str]): Additional instructions for tone, etc.
        
        Returns:
            dict: Contains status and error message (if any).
        """
        try:
            async with self.async_client.audio.speech.with_streaming_response.create(
                model=model,
                voice=voice,
                input=text,
                instructions=instructions,
                response_format=response_format
            ) as response:
                audio_data = await response.read()  # Read the audio data
                return {
                    "status": "success",
                    "audio_data": audio_data,
                    "content_type": f"audio/{response_format}"
                }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to generate audio: {str(e)}"
            }

    def get_supported_formats(self) -> list:
        """Return a list of supported audio formats."""
        return ["mp3", "opus", "aac", "flac", "wav", "pcm"]

    def get_supported_voices(self) -> list:
        """Return a list of supported voices."""
        return ["alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"]

# Example usage (for testing or integration)
if __name__ == "__main__":
    async def test_audio_service():
        audio_service = AudioService()
        
        # Test generating and saving audio
        result = await audio_service.generate_audio(
            text="Today is a wonderful day to build something people love!",
            voice="coral",
            instructions="Speak in a cheerful and positive tone.",
            response_format="mp3"
        )
        print(result)

        # Test streaming audio
        result = await audio_service.stream_audio(
            text="Hello, this is a test of streaming audio!",
            voice="alloy",
            instructions="Speak in a calm and clear tone.",
            response_format="mp3"
        )
        print(result)

    asyncio.run(test_audio_service())