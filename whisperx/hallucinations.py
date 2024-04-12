import re 

# TODO: 확장성을 고려한 더 좋은 구조르 고려해보도록 하자. 
# 이 방식대로라면 언어가 하나 추가 될때마다 LANGUAGE_CODES, ALLOWED_PATTERNS, HALLUCINATIONS 까지를 다 추가해줘야함. 
LANGUAGE_CODES = [
    "ko", "en", "ja", "zh", "es", "it", "ru", "id", "de", "nl"
]
# PATTERNS 
ALLOWED_PATTERNS = {
    "ko": re.compile("[가-힣a-zA-Z0-9\s\W]+"),
    "en": re.compile(".*"),
    "ja": re.compile(".*"),
    "zh": re.compile(".*"),
    "es": re.compile(".*"),
    "it": re.compile(".*"),
    "ru": re.compile(".*"),
    "id": re.compile(".*"),
    "de": re.compile(".*"),
    "nl": re.compile(".*")
}

HALLUCINATIONS = {
    "ko": [
        "이 시각 세계였습니다.",
        "MBC 뉴스 김성현입니다.",
        "지금까지 뉴스 스토리였습니다.",
        "시청해주셔서 감사합니다.",
        "날씨였습니다.",
        "자막 제공 배달의민족",
        "제작지원 자막 제작지원",
        "이 노래는 제가 부르는 노래입니다.",
        "아이유의 러브게임",
        "감사합니다.",
        "고맙습니다.",
        "아멘"
        ],
    "en": [],
    "ja": [],
    "zh": [],
    "es": [],
    "it": [],
    "ru": [],
    "id": [],
    "de": [],
    "nl": []
}


class HallucinationFilter:
    def __init__(self, language, allowed_patterns, hallucination_patterns ):
        self.language = language
        self.allowed_patterns = allowed_patterns 
        self.hallucination_patterns = hallucination_patterns
    
    def is_hallucination(self, text):
        if text in self.hallucination_patterns:
            return True
        return False
    
    def is_allowed_pattern(self, text):
        if re.match(self.allowed_patterns, text):
            return True
        return False
    
    def __call__(self, text):
        return self.is_hallucination(text) or not self.is_allowed_pattern(text)


hallucination_filters = {
    k: HallucinationFilter(
        language = k,
        allowed_patterns = ALLOWED_PATTERNS[k],
        hallucination_patterns = HALLUCINATIONS[k]
    ) for k in LANGUAGE_CODES
}
