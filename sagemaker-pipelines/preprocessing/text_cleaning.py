import re

emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002500-\U00002BEF"  # chinese char
    "\U00002702-\U000027B0"  # dingbats
    "\U00002702-\U000027B0"  # dingbats (duplicate, may be a mistake)
    "\U000024C2-\U0001F251"  # enclosed characters
    "\U0001f926-\U0001f937"  # supplemental symbols and pictographs
    "\U00010000-\U0010ffff"  # supplementary private use area-A
    "\u2640-\u2642"  # gender symbols
    "\u2600-\u2B55"  # miscellaneous symbols
    "\u200d"  # zero-width joiner
    "\u23cf"  # eject symbol
    "\u23e9"  # black right-pointing double triangle
    "\u231a"  # watch
    "\ufe0f"  # variation selector-16
    "\u3030"  # wavy dash
    "]+",
    flags=re.UNICODE,
)

# Additional patterns
special_char_pattern = re.compile(r"[^\w\s\|.,;:!?-]")
multiple_space_pattern = re.compile(r"\s+")


def process_text(text):
    text = emoji_pattern.sub(r"", text)
    text = special_char_pattern.sub(r"", text)
    text = multiple_space_pattern.sub(r" ", text)
    # text = text.lower()
    return text

