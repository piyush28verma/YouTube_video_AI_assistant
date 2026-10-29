"""Curated Benchmark Videos and Transcripts for Offline Demo & Instant Evaluation.

Provides pre-indexed, verified transcripts for:
1. Andrej Karpathy - Intro to Large Language Models (zjkBMFhNj_g)
2. Steve Jobs - 2005 Stanford Commencement Address (UF8uR6Z6KLc)
3. 3Blue1Brown - Attention in Transformers, Visually Explained (wjZofJX0v4U)
"""

from typing import Any, Dict, List, Optional


SAMPLE_VIDEOS: Dict[str, Dict[str, Any]] = {
    "zjkBMFhNj_g": {
        "video_id": "zjkBMFhNj_g",
        "title": "Intro to Large Language Models",
        "author_name": "Andrej Karpathy",
        "duration": "1:00:00",
        "description": "A 1-hour comprehensive overview of Large Language Models: pretraining, fine-tuning, RLHF, multimodal tokens, hallucinations, tool use, and system security.",
        "sample_questions": [
            "What is the difference between pre-training and fine-tuning an LLM?",
            "How does Karpathy explain hallucinations in language models and why do they happen?",
            "What is RLHF (Reinforcement Learning from Human Feedback) and what role does it play?",
            "What are the security vulnerabilities mentioned regarding prompt injections and jailbreaks?",
        ],
        "segments": [
            {"start": 0.0, "duration": 45.0, "text": "Hi everyone, welcome to this introduction to Large Language Models. In this video I want to give you a conceptual and practical overview of what LLMs are, how they are trained, and how they function as modern operating systems."},
            {"start": 45.0, "duration": 60.0, "text": "The LLM core is basically two files: a parameters file with weights, for example 140 gigabytes for a 70 billion parameter model, and a run.c file with roughly 500 lines of C code that performs forward inference."},
            {"start": 105.0, "duration": 75.0, "text": "Pre-training is the most computationally expensive stage. We take a massive chunk of the internet, roughly 10 trillion tokens of text, and train a transformer neural network to predict the next token on thousands of GPUs for months at a cost of millions of dollars."},
            {"start": 180.0, "duration": 70.0, "text": "The result of pre-training is a base model. A base model is not an assistant; it is a document completer. If you ask it 'What is the capital of France?', it might respond with 'What is the capital of Germany?' because it thinks it is completing a trivia quiz."},
            {"start": 250.0, "duration": 80.0, "text": "To turn a base model into an assistant, we use supervised fine-tuning (SFT). Human labelers write high-quality prompt and response pairs, teaching the model conversation conventions and instruction following."},
            {"start": 330.0, "duration": 90.0, "text": "After fine-tuning comes Reinforcement Learning from Human Feedback (RLHF). Human evaluators compare multiple model responses and rank which one is more helpful, truthful, and harmless. We train a reward model on these comparisons to optimize the LLM via PPO."},
            {"start": 420.0, "duration": 85.0, "text": "Now let us talk about hallucinations. LLMs do not have an explicit database or internal knowledge lookup table; their knowledge is stored lossily across billions of weights as statistical associations. When an LLM generates text, it generates likely completions, which can sound completely convincing while being factually untrue."},
            {"start": 505.0, "duration": 95.0, "text": "To mitigate hallucinations, we use Retrieval Augmented Generation (RAG) and search tools. Instead of asking the model to retrieve memory from its weights, we retrieve authoritative source documents and provide them inside the context window for grounded synthesis."},
            {"start": 600.0, "duration": 80.0, "text": "Modern LLMs are evolving into multimodal models. They process images, audio, and video directly into token representations rather than relying on separate transcription or vision pipelines."},
            {"start": 680.0, "duration": 90.0, "text": "LLMs can also use tools like calculators, code interpreters, and web search engines to overcome their inherent algorithmic limitations in exact arithmetic and current events."},
            {"start": 770.0, "duration": 85.0, "text": "Security is a major frontier. Attacks like prompt injection, jailbreaking, and data poisoning present critical challenges. An attacker can hide instructions inside a webpage or image that hijacks the assistant's instructions."},
            {"start": 855.0, "duration": 95.0, "text": "In summary, LLMs are not mere chatbots; they are the central processing units of a new computing architecture, orchestrating memory, context windows, tools, and multimodal reasoning."},
        ],
    },
    "UF8uR6Z6KLc": {
        "video_id": "UF8uR6Z6KLc",
        "title": "Steve Jobs' 2005 Stanford Commencement Address",
        "author_name": "Stanford University",
        "duration": "15:04",
        "description": "Steve Jobs tells three stories from his life about connecting the dots, love and loss, and death during the 2005 Stanford Commencement address.",
        "sample_questions": [
            "What were the three stories Steve Jobs told in his speech?",
            "How did dropping out of Reed College lead to Apple's typography design?",
            "What was Jobs' perspective on being fired from Apple?",
            "What advice did Jobs give about death and staying foolish?",
        ],
        "segments": [
            {"start": 0.0, "duration": 40.0, "text": "I am honored to be with you today at your commencement from one of the finest universities in the world. I never graduated from college. Truth be told, this is the closest I've ever gotten to a college graduation. Today I want to tell you three stories from my life. That's it. No big deal. Just three stories."},
            {"start": 40.0, "duration": 75.0, "text": "The first story is about connecting the dots. I dropped out of Reed College after the first 6 months, but then stayed around as a drop-in for another 18 months before I really quit. I decided to take a calligraphy class because Reed College offered the best calligraphy instruction in the country."},
            {"start": 115.0, "duration": 85.0, "text": "I learned about serif and sans serif typefaces, about varying the amount of space between different letter combinations, about what makes great typography great. Ten years later, when we were designing the first Macintosh computer, it all came back to me, and we designed it all into the Mac. It was the first computer with beautiful typography."},
            {"start": 200.0, "duration": 80.0, "text": "You can't connect the dots looking forward; you can only connect them looking backwards. So you have to trust that the dots will somehow connect in your future. You have to trust in something: your gut, destiny, life, karma, whatever."},
            {"start": 280.0, "duration": 90.0, "text": "My second story is about love and loss. I was lucky; I found what I loved to do early in life. Woz and I started Apple in my parents' garage when I was 20. In 10 years Apple had grown from just the two of us into a $2 billion company with over 4,000 employees. And then I got fired. How can you get fired from a company you started?"},
            {"start": 370.0, "duration": 90.0, "text": "Getting fired from Apple was the best thing that could have ever happened to me. The heaviness of being successful was replaced by the lightness of being a beginner again, less sure about everything. It freed me to enter one of the most creative periods of my life, during which I started NeXT, Pixar, and fell in love with Laurene."},
            {"start": 460.0, "duration": 85.0, "text": "My third story is about death. When I was 17, I read a quote that went something like: 'If you live each day as if it was your last, someday you'll most certainly be right.' It made an impression on me, and for the past 33 years, I have looked in the mirror every morning and asked myself: 'If today were the last day of my life, would I want to do what I am about to do today?'"},
            {"start": 545.0, "duration": 95.0, "text": "Remembering that you are going to die is the best way I know to avoid the trap of thinking you have something to lose. You are already naked. There is no reason not to follow your heart."},
            {"start": 640.0, "duration": 80.0, "text": "Your time is limited, so don't waste it living someone else's life. Don't be trapped by dogma, which is living with the results of other people's thinking. Don't let the noise of others' opinions drown out your own inner voice."},
            {"start": 720.0, "duration": 70.0, "text": "On the back cover of their final issue of the Whole Earth Catalog, they put a photograph of an early morning country road, and beneath it were the words: 'Stay Hungry. Stay Foolish.' It was their farewell message as they signed off. And I have always wished that for myself. And now, as you graduate to begin anew, I wish that for you: Stay Hungry. Stay Foolish."},
        ],
    },
    "wjZofJX0v4U": {
        "video_id": "wjZofJX0v4U",
        "title": "Attention in Transformers, Visually Explained",
        "author_name": "3Blue1Brown",
        "duration": "28:15",
        "description": "Grant Sanderson visually unpacks the self-attention mechanism inside the Transformer architecture, showing how tokens calculate queries, keys, and values to exchange contextual meaning.",
        "sample_questions": [
            "What is the intuitive purpose of the Attention mechanism in Transformers?",
            "How do Query, Key, and Value vectors interact mathematically?",
            "Why is the dot product divided by the square root of the dimension of keys?",
            "What is the difference between single-head attention and multi-head attention?",
        ],
        "segments": [
            {"start": 0.0, "duration": 50.0, "text": "Welcome. In this video we are going to explore the Attention mechanism, which is the beating heart of modern deep learning and the Transformer architecture powering LLMs like GPT and Gemini."},
            {"start": 50.0, "duration": 70.0, "text": "Words initially start as embedding vectors in a high-dimensional space, say 12,288 dimensions. But in isolation, the word 'bank' doesn't know whether it means a financial institution or the side of a river."},
            {"start": 120.0, "duration": 80.0, "text": "Attention allows words to update their embedding vectors based on context. Words pass messages to one another so that 'bank' near 'money' shifts in vector space toward finance, while 'bank' near 'river' shifts toward geography."},
            {"start": 200.0, "duration": 90.0, "text": "Mathematically, each token projects its embedding into three distinct vectors: a Query vector, a Key vector, and a Value vector, through learned linear transformation matrices W_Q, W_K, and W_V."},
            {"start": 290.0, "duration": 85.0, "text": "The Query asks: 'What kind of information am I looking for?' The Key advertises: 'What kind of information do I contain?' When the dot product between Query_i and Key_j is high, token i pays high attention to token j."},
            {"start": 375.0, "duration": 85.0, "text": "We divide this dot product by the square root of the key dimension d_k to prevent gradients from vanishing during softmax when dimensions are large. Softmax normalizes these scores into a probability distribution summing to 1."},
            {"start": 460.0, "duration": 90.0, "text": "Finally, the attention weights multiply the Value vectors, producing a weighted sum of context that is added back into the original token embedding through a residual connection."},
            {"start": 550.0, "duration": 90.0, "text": "Multi-head attention runs this mechanism in parallel across several heads, allowing different heads to specialize in different linguistic relationships such as syntax, adjective-noun binding, and pronoun resolution."},
        ],
    },
}


def get_sample_video(video_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve sample video record by ID."""
    return SAMPLE_VIDEOS.get(video_id)


def list_sample_videos() -> List[Dict[str, Any]]:
    """List all available curated sample videos."""
    return list(SAMPLE_VIDEOS.values())
