# LangChain Pinecone Mistral LLM RAG Astrology Chatbot

## Overview

Welcome to the LangChain Pinecone Mistral LLM RAG Astrology Chatbot! This project integrates the latest advancements in language models and retrieval-augmented generation to provide personalized astrology readings. The chatbot leverages LangChain, Pinecone, and the Mistral 7B model from Hugging Face, all hosted on Streamlit.

## Features

- **Interactive Astrology Readings**: Get personalized astrological insights.
- **Advanced AI Models**: Utilizes Mistral 7B model for accurate and intelligent responses.
- **Real-time Interaction**: Deployed on Streamlit for seamless user experience.
- **Scalable and Efficient**: Powered by Pinecone for efficient data retrieval.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/astrology-chatbot.git
    cd astrology-chatbot
    ```

2. Set up a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up environment variables:
    Create a `.env` file in the root directory and add your Pinecone API key and Hugging Face API key:
    ```env
    PINECONE_API_KEY=your_pinecone_api_key
    HUGGINGFACE_API_KEY=your_huggingface_api_key
    ```

## Usage

1. Run the Streamlit app:
    ```sh
    streamlit run index.py
    ```

2. Open your browser and navigate to the URL provided by Streamlit to interact with the chatbot.

## Components

- **LangChain**: Framework for building language models.
- **Pinecone**: Vector database for fast and scalable data retrieval.
- **Hugging Face Mistral 7B**: Language model for generating responses.
- **Streamlit**: Web application framework for interactive user interface.

## Future Improvements

- Add more horoscope data for other signs.
- Implement a feedback mechanism to improve response accuracy.
- Explore integration with additional NLP models.

## Contributing

We welcome contributions! Feel free to fork the repository and submit pull requests.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

For any questions or feedback, please contact [Ayush12122003@gmail.com].
