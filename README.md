# Video Analysis with OpenAI Vision

A Streamlit application that analyzes videos using OpenAI's GPT-4o model with vision capabilities.

## Features

- Upload video files in various formats (MP4, MOV, AVI, MKV)
- Customize system and client prompts
- Adjust temperature and other model parameters
- Extract and analyze frames from the video
- Get detailed analysis from OpenAI's vision-capable models

## Installation

1. Clone this repository:
```bash
git clone https://github.com/viggneshk/videoanalysis.git
cd videoanalysis
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
   - Copy the `.env.example` file to `.env`
   - Add your OpenAI API key to the `.env` file

```bash
cp .env.example .env
# Edit the .env file and add your API key
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Open your browser and go to `http://localhost:8501`

3. Use the interface to:
   - Upload a video file
   - Customize your prompts and parameters
   - Click "Analyze Video" to process the video

## Requirements

- Python 3.9+
- OpenAI API key with access to GPT-4o model
- Sufficient API credits for video processing

## Notes

- The application extracts a limited number of frames to stay within OpenAI's API limits
- Processing time and costs depend on video length and the number of frames analyzed
- API usage will incur charges on your OpenAI account

## License

MIT
