# LinkedIn Post Generator ✍️

An AI-powered LinkedIn post generator built with Streamlit, Groq LLMs, semantic example retrieval, and few-shot prompting.

## Features

- Generate LinkedIn posts by topic
- Choose post length (Short, Medium, Long)
- Generate posts in English or Hinglish
- Select writing tone:
  - Storytelling
  - Educational
  - Motivational
  - Controversial / Hot Take
  - Witty / Humorous
- Choose content structure templates
- Semantic retrieval of similar posts for style guidance
- Embedding caching for faster startup
- Offline TF-IDF fallback when Sentence Transformers are unavailable

---

## Project Structure

```text
postgen_upgraded/
│
├── main.py                 # Streamlit UI
├── post_generator.py       # Prompt construction and generation logic
├── few_shot.py             # Semantic retrieval engine
├── preprocess.py           # Metadata extraction and tag normalization
├── llm_helper.py           # Groq LLM configuration
├── requirements.txt
│
├── data/
│   ├── raw_post.json
│   ├── prerocessed_post.json
│   ├── expanded_posts.json
│   └── embeddings_cache.pkl
```

---

## How It Works

### 1. Dataset Preparation

Raw LinkedIn posts are stored in JSON format.

`preprocess.py` uses an LLM to extract:

- Line count
- Language
- Tags

It then normalizes similar tags into a unified taxonomy.

Example:

```text
Jobseekers -> Job Search
Job Hunting -> Job Search
Motivation -> Motivation
Inspiration -> Motivation
```

---

### 2. Semantic Few-Shot Retrieval

`few_shot.py` loads the processed posts and creates embeddings.

Embedding strategy:

1. Try SentenceTransformer (`all-MiniLM-L6-v2`)
2. If unavailable, fall back to TF-IDF vectors

Retrieved examples are ranked using cosine similarity and filtered by:

- Language
- Length
- Topic tag

These examples are injected into prompts to mimic writing style without copying content.

---

### 3. Prompt Engineering

`post_generator.py` builds prompts using:

- Topic
- Length
- Language
- Tone
- Template structure
- Retrieved examples

Supported templates:

- Free Form
- Hook → Problem → Solution → CTA
- Listicle
- Story Arc
- Controversial Opinion

---

### 4. LLM Generation

The application uses Groq through LangChain.

Current model:

```python
llama-3.3-70b-versatile
```

Configuration is located in `llm_helper.py`.

---

## Installation

### Clone Repository

```bash
git clone <your-repository-url>
cd postgen_upgraded
```

### Create Virtual Environment with uv

```bash
uv venv
```

Activate:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
uv pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key
```

---

## Run the Application

```bash
streamlit run main.py
```

Open the local URL displayed by Streamlit.

---

## Example Workflow

1. Select a topic.
2. Choose language.
3. Select desired length.
4. Pick a tone.
5. Select a structure template.
6. Click **Generate Post**.
7. Copy the generated LinkedIn post.

---

## Technologies Used

- Python
- Streamlit
- LangChain
- Groq API
- Sentence Transformers
- Scikit-learn
- Pandas
- NumPy

---

## Future Improvements

- Multi-language support
- User-defined custom tones
- Hashtag generation
- Post scheduling integration
- FAISS vector search
- Analytics and engagement suggestions

---

## License

This project is intended for educational and learning purposes. Update the license section as needed for your repository.
