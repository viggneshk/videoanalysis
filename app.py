import os
import streamlit as st
import tempfile
import cv2
import numpy as np
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
from moviepy.editor import VideoFileClip
import base64
import requests
import json

# Load environment variables
load_dotenv()

def extract_frames(video_path, max_frames=20, extract_all=False):
    """Extract frames from a video file"""
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    # Calculate frame interval to extract frames evenly
    interval = 1 if extract_all else max(1, total_frames // max_frames)
    
    frames = []
    frame_positions = []
    
    # Create a progress bar for frame extraction
    if extract_all:
        # Only show progress bar when extracting all frames
        expected_frames = min(total_frames, total_frames // interval + 1)
        progress_bar = st.progress(0)
    
    for i in range(0, total_frames, interval):
        if not extract_all and len(frames) >= max_frames:
            break
        
        # Update progress bar if extracting all frames
        if extract_all and i % 10 == 0:
            progress = min(1.0, len(frames) / expected_frames)
            progress_bar.progress(progress)
        
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, frame = video.read()
        
        if success:
            # Convert from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            frame_positions.append(i / fps)  # Time in seconds
    
    # Complete the progress bar
    if extract_all:
        progress_bar.progress(1.0)
        
    video.release()
    return frames, frame_positions, duration

def encode_image_to_base64(image_array):
    """Convert numpy array to base64 string"""
    pil_image = Image.fromarray(image_array)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        pil_image.save(temp_file.name)
        with open(temp_file.name, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
        
def analyze_video(frames, system_prompt, client_prompt, temperature, api_key, max_api_frames=50, model="gpt-4o"):
    """Analyze video frames using OpenAI's API"""
    
    # Ensure we have a valid API key
    if not api_key or not api_key.startswith("sk-"):
        raise ValueError("Invalid OpenAI API key. Please check your API key configuration.")
    
    # Limit the number of frames sent to the API
    if len(frames) > max_api_frames:
        # Sample frames evenly
        step = len(frames) // max_api_frames
        api_frames = frames[::step][:max_api_frames]
    else:
        api_frames = frames
    
    # Prepare frames for API
    base64_frames = [encode_image_to_base64(frame) for frame in api_frames]
    
    # Prepare the content for the API call
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": client_prompt},
                *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_frame}", "detail": "auto"}} 
                  for base64_frame in base64_frames]
            ],
        },
    ]
    
    # Try using OpenAI SDK first
    try:
        # Set API key in environment variable
        os.environ["OPENAI_API_KEY"] = api_key
        
        # Create a fresh client
        client = OpenAI()
        
        # Make the API call
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=4000,
        )
        
        return response.choices[0].message.content
    except Exception as e:
        # If SDK fails, try direct API call with requests
        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 4000
            }
            
            response = requests.post(url, headers=headers, json=payload)
            response_data = response.json()
            
            if "choices" in response_data and len(response_data["choices"]) > 0:
                return response_data["choices"][0]["message"]["content"]
            else:
                raise Exception(f"API error: {response_data}")
        except Exception as req_error:
            # If both methods fail, raise the original error
            raise Exception(f"Failed to connect to OpenAI API: {str(e)}\nAttempted direct API call but got: {str(req_error)}")

def estimate_cost(num_frames, avg_tokens_per_frame=765, output_tokens=1000, model="gpt-4o"):
    """Estimate the cost of analyzing frames with OpenAI's API"""
    # Model pricing (as of current rates)
    pricing = {
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "o1": {"input": 0.018, "output": 0.06}
    }
    
    if model not in pricing:
        model = "gpt-4o"  # Default fallback
        
    input_cost_per_1k = pricing[model]["input"]
    output_cost_per_1k = pricing[model]["output"]
    
    total_input_tokens = num_frames * avg_tokens_per_frame
    input_cost = (total_input_tokens / 1000) * input_cost_per_1k
    output_cost = (output_tokens / 1000) * output_cost_per_1k
    
    return input_cost + output_cost, total_input_tokens + output_tokens

def get_model_info(model_name):
    """Get information about the selected model"""
    models = {
        "gpt-4o": {
            "description": "Fast, intelligent, flexible GPT model with vision capabilities",
            "context_window": "128,000 tokens",
            "strengths": "Good balance of performance and cost",
            "pricing_note": "$0.005/1K input tokens, $0.015/1K output tokens"
        },
        "gpt-4o-mini": {
            "description": "Fast, flexible, intelligent reasoning model with vision capabilities",
            "context_window": "128,000 tokens",
            "strengths": "Very affordable for most tasks",
            "pricing_note": "$0.00015/1K input tokens, $0.0006/1K output tokens"
        },
        "gpt-4-turbo": {
            "description": "GPT-4 Turbo with vision capabilities",
            "context_window": "128,000 tokens",
            "strengths": "High quality visual reasoning",
            "pricing_note": "$0.01/1K input tokens, $0.03/1K output tokens"
        },
        "gpt-4": {
            "description": "Original GPT-4 with vision capabilities",
            "context_window": "8,000 tokens",
            "strengths": "High quality instruction following",
            "pricing_note": "$0.03/1K input tokens, $0.06/1K output tokens"
        },
        "o1": {
            "description": "High intelligence reasoning model with vision capabilities",
            "context_window": "128,000 tokens",
            "strengths": "Advanced reasoning and intelligence",
            "pricing_note": "$0.018/1K input tokens, $0.06/1K output tokens"
        }
    }
    
    return models.get(model_name, {
        "description": "OpenAI model with vision capabilities",
        "context_window": "Unknown",
        "strengths": "Various capabilities",
        "pricing_note": "Pricing varies"
    })

def main():
    st.set_page_config(page_title="Video Analysis with OpenAI", page_icon="🎬", layout="wide")
    
    # Initialize session state for tracking values
    if 'max_frames' not in st.session_state:
        st.session_state.max_frames = 10
    if 'extract_all_frames' not in st.session_state:
        st.session_state.extract_all_frames = False
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.7
    if 'model' not in st.session_state:
        st.session_state.model = "gpt-4o"
    if 'show_comparison' not in st.session_state:
        st.session_state.show_comparison = False
    if 'api_key' not in st.session_state:
        # Try to get API key from Streamlit secrets first
        try:
            if hasattr(st, 'secrets') and 'openai' in st.secrets and 'OPENAI_API_KEY' in st.secrets.openai:
                st.session_state.api_key = st.secrets.openai.OPENAI_API_KEY
            else:
                st.session_state.api_key = ""
        except:
            st.session_state.api_key = ""
    
    # Apply global CSS for a more compact layout
    st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1000px;
        margin: 0 auto;
    }
    .stVideo {
        max-width: 500px !important;
        margin: 0 auto;
    }
    .st-emotion-cache-1kyxreq {
        max-width: 100%;
        margin: 0 auto;
    }
    .st-emotion-cache-16txtl3 h1 {
        margin-bottom: 0.5rem;
    }
    .stTextArea label p {
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("Video Analysis with OpenAI Vision")
    st.write("Upload a video to analyze it using OpenAI's GPT-4o Vision model.")
    
    # Check if API key already exists in session state or secrets
    api_key_exists = st.session_state.api_key != ""
    
    # Only show API key input if not already available
    if not api_key_exists:
        # Simple API key input at the top
        api_key = st.text_input(
            "OpenAI API Key", 
            type="password",
            value=st.session_state.api_key,
            help="Your API key will be stored in your session and not saved on our servers."
        )
        
        # Save the API key to session state if it's valid
        if api_key:
            if api_key.startswith("sk-"):
                st.session_state.api_key = api_key
            else:
                st.error("Please enter a valid OpenAI API key starting with 'sk-'")
                st.session_state.api_key = ""
                
        # Check if API key is available
        if not st.session_state.api_key:
            st.warning("⚠️ Please enter your OpenAI API key to use this app.")
            st.info("""
            You need an OpenAI API key with access to GPT-4 Vision models.
            Get your API key from [OpenAI's website](https://platform.openai.com/api-keys).
            
            Your API key will only be used for this session and is not stored on our servers.
            """)
            return
    else:
        # Show a success message that API key is configured
        st.success("✅ OpenAI API key configured successfully.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"])
    
    # Cost estimation container
    cost_container = st.empty()
    
    # Model comparison toggle
    show_comparison = st.checkbox("Show Model Comparison Table", value=st.session_state.show_comparison, 
                                 key="comparison_checkbox",
                                 on_change=lambda: setattr(st.session_state, 'show_comparison', 
                                                         st.session_state.comparison_checkbox))
    
    if show_comparison:
        st.markdown("""
        ### OpenAI Vision Model Comparison
        
        | Model | Description | Context Window | Relative Cost | Best For |
        |-------|-------------|----------------|--------------|----------|
        | **GPT-4o-mini** | Fast, lightweight model | 128K tokens | $ (Lowest) | Quick, affordable analysis |
        | **GPT-4o** | Balanced performance | 128K tokens | $$ | General purpose analysis |
        | **GPT-4-turbo** | Higher quality | 128K tokens | $$$ | Detailed video analysis |
        | **GPT-4** | Original GPT-4 | 8K tokens | $$$$ | High quality, shorter videos |
        | **o1** | Advanced reasoning | 128K tokens | $$$$$ | Complex analysis tasks |
        
        *Note: Costs are relative and actual costs will depend on video length and complexity.*
        """)
    
    # Real-time parameter controls
    st.subheader("Analysis Parameters")
    param_cols = st.columns(4)
    
    with param_cols[0]:
        model = st.selectbox(
            "Model",
            options=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "o1"],
            index=0,
            key="model_selector",
            on_change=lambda: setattr(st.session_state, 'model', st.session_state.model_selector),
            help="Select which OpenAI vision model to use"
        )
    
    with param_cols[1]:
        temperature = st.slider("Temperature", 
                               min_value=0.0, max_value=2.0, 
                               value=st.session_state.temperature, 
                               step=0.1,
                               key="temp_slider",
                               on_change=lambda: setattr(st.session_state, 'temperature', st.session_state.temp_slider))
    
    with param_cols[2]:
        max_frames = st.slider("Max Frames to Analyze", 
                              min_value=5, max_value=50, 
                              value=st.session_state.max_frames,
                              step=1,
                              key="frames_slider",
                              on_change=lambda: setattr(st.session_state, 'max_frames', st.session_state.frames_slider))
    
    with param_cols[3]:
        extract_all_frames = st.checkbox("Extract All Frames", 
                                       value=st.session_state.extract_all_frames,
                                       key="extract_all_checkbox",
                                       on_change=lambda: setattr(st.session_state, 'extract_all_frames', st.session_state.extract_all_checkbox),
                                       help="Warning: This may cause performance issues with longer videos")
    
    # Real-time cost estimation
    frames_estimate = max_frames if not extract_all_frames else 50  # Assume worst case for extract_all
    estimated_cost, estimated_tokens = estimate_cost(frames_estimate, model=model)
    
    # Show model information
    model_info = get_model_info(model)
    with st.expander("Model Information"):
        st.markdown(f"""
        **{model.upper()}**: {model_info['description']}
        
        - **Context Window**: {model_info['context_window']}
        - **Strengths**: {model_info['strengths']}
        - **Pricing**: {model_info['pricing_note']}
        """)
        
        # Add cost comparison for current frame count
        st.markdown("#### Cost Comparison For Current Settings")
        
        # Calculate costs for all models with current settings
        models = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4", "o1"]
        costs = []
        for m in models:
            cost, _ = estimate_cost(frames_estimate, model=m)
            costs.append(cost)
        
        # Create a simple bar chart using Markdown
        max_cost = max(costs)
        bar_lengths = [int(cost/max_cost * 20) for cost in costs]
        
        for i, m in enumerate(models):
            bar = "█" * bar_lengths[i] + "░" * (20 - bar_lengths[i])
            st.markdown(f"**{m}**: ${costs[i]:.2f} {bar}")
    
    cost_container.info(f"💰 Estimated cost with current settings: ${estimated_cost:.2f} ({estimated_tokens:,} tokens)")
    
    # Form for prompts
    with st.form(key="analysis_form"):
        # Create two columns for the form
        col1, col2 = st.columns(2)
        
        with col1:
            system_prompt = st.text_area(
                "System Prompt",
                value="You are a video analysis assistant powered by GPT-4o. Analyze the frames from the video and provide detailed insights.",
                height=100
            )
            
        with col2:
            client_prompt = st.text_area(
                "Client Prompt", 
                value="Please analyze this video and describe what's happening. Focus on key events, objects, and actions.",
                height=100
            )
        
        submit_button = st.form_submit_button(label="Analyze Video")
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(uploaded_file.getvalue())
            video_path = temp_file.name
            
        # Display video
        st.video(video_path, start_time=0)
        
        # When the user clicks the analyze button
        if submit_button:
            with st.spinner('Extracting frames from the video...'):
                frames, timestamps, duration = extract_frames(video_path, max_frames=max_frames, extract_all=extract_all_frames)
                
                # Display some extracted frames
                st.write(f"Extracted {len(frames)} frames from a {duration:.2f} second video")
                
                # Display a subset of frames in a more compact way
                st.markdown("""
                <style>
                .frame-container img {
                    max-height: 120px !important;
                    width: auto !important;
                    object-fit: contain;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Display all frames
                st.markdown('<div class="frame-container">', unsafe_allow_html=True)
                
                # Calculate number of frames per row - more frames means smaller columns
                if len(frames) <= 10:
                    frames_per_row = min(5, len(frames))
                elif len(frames) <= 20:
                    frames_per_row = min(8, len(frames))
                else:
                    frames_per_row = min(10, len(frames))
                
                # Display all frames in multiple rows
                for i in range(0, len(frames), frames_per_row):
                    cols = st.columns(frames_per_row)
                    for j, col in enumerate(cols):
                        if i + j < len(frames):
                            frame_idx = i + j
                            with col:
                                st.image(frames[frame_idx], 
                                        caption=f"Frame at {timestamps[frame_idx]:.2f}s", 
                                        use_column_width=False, 
                                        width=100 if len(frames) > 20 else 120)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with st.spinner('Analyzing video with OpenAI...'):
                try:
                    # Add a warning for large numbers of frames
                    max_api_frames = 50
                    frames_to_analyze = min(len(frames), max_api_frames)
                    
                    # Display cost estimate
                    estimated_cost, estimated_tokens = estimate_cost(frames_to_analyze, model=model)
                    st.info(f"💰 Estimated cost: ${estimated_cost:.2f} (approximately {estimated_tokens:,} tokens)")
                    
                    if len(frames) > max_api_frames:
                        st.warning(f"⚠️ {len(frames)} frames were extracted, but only {max_api_frames} will be sent to OpenAI API to stay within limits.")
                    
                    # Use the API key from session state
                    analysis_result = analyze_video(
                        frames, 
                        system_prompt, 
                        client_prompt, 
                        temperature, 
                        api_key=st.session_state.api_key, 
                        max_api_frames=max_api_frames, 
                        model=model
                    )
                    
                    # Display results
                    st.success("Analysis complete!")
                    st.markdown("## Analysis Results")
                    st.markdown(analysis_result)
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    
        # Clean up the temporary file
        try:
            os.unlink(video_path)
        except:
            pass

if __name__ == "__main__":
    main() 