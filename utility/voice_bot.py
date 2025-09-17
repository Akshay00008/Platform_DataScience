import assemblyai as aai
import openai
import elevenlabs
from queue import Queue

# Set API keys
aai.settings.api_key = "API-KEY"
openai.api_key = "API-KEY"
elevenlabs.set_api_key("API-KEY")

# Initialize a queue to hold transcripts
transcript_queue = Queue()

def on_data(transcript: aai.RealtimeTranscript):
    """
    Callback to handle the real-time transcript data.
    """
    if not transcript.text:
        return
    if isinstance(transcript, aai.RealtimeFinalTranscript):
        transcript_queue.put(transcript.text + '')
        print("User:", transcript.text, end="\r\n")
    else:
        print(transcript.text, end="\r")

def on_error(error: aai.RealtimeError):
    """
    Callback to handle errors during transcription.
    """
    print("An error occurred:", error)

def handle_conversation():
    """
    Main conversation loop that handles real-time transcription,
    communicates with OpenAI, and converts AI responses to audio.
    """
    while True:
        transcriber = aai.RealtimeTranscriber(
            on_data=on_data,
            on_error=on_error,
            sample_rate=44_100,
        )

        # Start the connection
        transcriber.connect()

        # Open the microphone stream
        microphone_stream = aai.extras.MicrophoneStream()

        # Stream audio from the microphone
        transcriber.stream(microphone_stream)

        # Close the transcription session
        transcriber.close()

        # Retrieve the transcript from the queue
        transcript_result = transcript_queue.get()

        # Send the transcript to OpenAI for response generation
        response = openai.ChatCompletion.create(
            model='gpt-4',
            messages=[
                {"role": "system", "content": 'You are a highly skilled AI, answer the questions given within a maximum of 1000 characters.'},
                {"role": "user", "content": transcript_result}
            ]
        )

        # Extract the AI response text
        text = response['choices'][0]['message']['content']

        # Convert the response to audio and play it
        audio = elevenlabs.generate(
            text=text,
            voice="Bella"  # You can change this to any voice of your choice
        )

        print("\nAI:", text, end="\r\n")

        # Play the generated audio response
        elevenlabs.play(audio)

def main():
    """
    Entry point for the script. Initializes and runs the conversation handler.
    """
    handle_conversation()

if __name__ == "__main__":
    main()
