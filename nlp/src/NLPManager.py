import re
from typing import Dict

class NLPManager:
    def __init__(self):
        pass

    def words_to_numbers(self, words: str) -> str:
        """Convert spelled-out numbers to digits. This is useful for extracting and normalizing numeric data from text."""
        word_map = {
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "niner": "9"
        }
        return ''.join(word_map.get(word.lower(), '') for word in words.split())


#USE THIS ONE
    def extract_info(self, text: str) -> Dict[str, str]:
        """Extract information using regular expressions, capturing target, heading, and tool from text.
        
        The extraction is robust against variations in ordering of the phrases in the input text.
        """
        patterns = {
            #"heading": r"heading is (\w+ \w+ \w+|\d{3})(?:,| tool to deploy is| target is|$)",
            "heading": r"heading is (.*?)(?:, tool to deploy is|, target is|$)"
,
            "target": r"target is (.*?)(?:, tool to deploy is|, heading is|$)",
            "tool": r"tool to deploy is ([^,]+?)(?:,| target is| heading is|$)"
        }

        info = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                if key == "heading":
                    # Convert any spelled-out numbers to numeric form, if applicable
                    info[key] = self.words_to_numbers(extracted) if not extracted.isdigit() else extracted
                else:
                    info[key] = extracted
            else:
                info[key] = None  # Fill missing fields with None for consistency
        
        return info
    

    def qa(self, context: str) -> Dict[str, str]:
        """Process a context string to extract question-answer style information based on predefined patterns."""
        return self.extract_info(context)
