
ENGLISH_PROMPTS = [
    # "The previous text agrees with the following: {target}"
    'Interviewer: "{target}" Politician: "{context}"'
]

GERMAN_PROMPTS = [
#    "Der bisherige Text stimmt mit Folgendem überein : {target}" 
    'Interviewer: "{target}" Politiker: "{context}"'
]

FRENCH_PROMPTS = [
#    "Le texte précédent concorde avec ce qui suit : {target}"
  'Intervieweur: "{target}" Politicien: "{context}" '
]


PROMPT_MAP = {
    "en": ENGLISH_PROMPTS,
    "de": GERMAN_PROMPTS,
    "fr": FRENCH_PROMPTS
}


GERMAN_HYPS = [
    "Der Politiker stimmt dem Interviewer zu."
]
FRENCH_HYPS = [
    "Le politicien était d'accord avec l'intervieweur."
]

HYP_MAP = {
    "de": GERMAN_HYPS,
    "fr": FRENCH_HYPS
}