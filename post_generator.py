from llm_helper import llm
from few_shot import FewShotPosts

fs = FewShotPosts()

# ------------------------------------------------------------------
# Tone definitions — label shown in UI + instruction injected in prompt
# ------------------------------------------------------------------
TONE_INSTRUCTIONS = {
    "Storytelling": (
        "Write in a personal narrative style. Start with a specific moment or event. "
        "Use 'I' voice. Build emotional arc: situation → conflict/realization → lesson."
    ),
    "Educational": (
        "Write in a clear, structured teaching style. Use numbered lists or short paragraphs. "
        "Include a concrete fact, statistic, or actionable insight. End with a takeaway."
    ),
    "Controversial / Hot Take": (
        "Start with a bold, counter-intuitive statement that challenges conventional wisdom. "
        "Back it up with 2-3 tight reasons. Don't hedge. Own the opinion."
    ),
    "Motivational": (
        "Write in an uplifting, empathetic tone. Acknowledge struggle before offering hope. "
        "Use short punchy sentences. End with an encouraging call to action."
    ),
    "Witty / Humorous": (
        "Use dry wit and light sarcasm. Subvert expectations. Keep it punchy. "
        "The humour should feel like something a sharp professional would say, not a stand-up comedian."
    ),
}

# ------------------------------------------------------------------
# Template structures — label shown in UI + scaffold injected in prompt
# ------------------------------------------------------------------
TEMPLATE_STRUCTURES = {
    "None (Free form)": "",
    "Hook → Problem → Solution → CTA": (
        "Structure the post as follows:\n"
        "1. Hook (1-2 lines): grab attention immediately\n"
        "2. Problem (2-3 lines): describe the pain point vividly\n"
        "3. Solution (3-4 lines): offer the insight or fix\n"
        "4. CTA (1 line): tell the reader what to do next (save, comment, share, reflect)"
    ),
    "Listicle (X things I learned...)": (
        "Structure the post as a numbered list:\n"
        "- Start with an engaging title line (e.g. '5 things nobody tells you about X:')\n"
        "- List 4-6 short, punchy items\n"
        "- End with one closing thought or invitation to discuss"
    ),
    "Story Arc (Struggle → Turning Point → Lesson)": (
        "Structure the post as a 3-act mini-story:\n"
        "1. Struggle: set the scene — what was going wrong?\n"
        "2. Turning point: what changed? (a decision, conversation, or realization)\n"
        "3. Lesson: what's the takeaway for the reader?"
    ),
    "Controversial Opinion": (
        "Structure the post as an opinion piece:\n"
        "- Line 1: the bold, provocative claim\n"
        "- Lines 2-5: 3-4 supporting reasons or examples\n"
        "- Final line: a question or challenge to the reader"
    ),
}


def get_length_str(length):
    if length == "Short":
        return "1 to 5 lines"
    if length == "Medium":
        return "6 to 10 lines"
    if length == "Long":
        return "11 to 15 lines"


def get_prompt(topic, length, language, tone, template):
    length_str = get_length_str(length)
    tone_instruction = TONE_INSTRUCTIONS.get(tone, "")
    template_instruction = TEMPLATE_STRUCTURES.get(template, "")

    prompt = f"""Generate a LinkedIn post using the below information. No preamble.

1) Topic: {topic}
2) Length: {length_str}
3) Language: {language}
   If Language is Hinglish it means a mix of Hindi and English. Script should always be English.
4) Tone: {tone}
   {tone_instruction}
"""

    if template_instruction:
        prompt += f"\n5) Structure to follow:\n{template_instruction}\n"

    examples = fs.get_filtered_posts(length, language, topic)
    if examples:
        prompt += "\n6) Use the writing style from these examples (do NOT copy content):\n"
        for i, post in enumerate(examples[:2]):
            prompt += f"\nExample {i + 1}:\n{post['text']}\n"

    return prompt


def generate_post(topic, length, language, tone, template):
    prompt = get_prompt(topic, length, language, tone, template)
    response = llm.invoke(prompt)
    return response.content
