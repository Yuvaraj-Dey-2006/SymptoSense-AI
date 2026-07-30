"""
augment_intents.py
Adds extra paraphrased patterns to each tag in intents.json so the
classifier generalizes to phrasing it hasn't seen exactly, instead
of only matching near-identical strings.

Run this ONCE, before train_model.py:
    python augment_intents.py
It reads intents.json and overwrites it with the augmented version
(a backup is saved as intents_backup.json first).
"""

import json
import shutil

INPUT_FILE = "intents.json"
BACKUP_FILE = "intents_backup.json"

# Extra patterns to append to each existing tag
EXTRA_PATTERNS = {
    "greeting": ["Hi there", "Good morning", "Hey Assistant", "Yo"],
    "send_off": ["Thanks a lot", "Bye", "Thank you so much", "That's all, thanks"],

    "cancer_general": ["Can you explain cancer to me?", "I want to know about cancer", "What causes cancer?"],
    "cancer_types": ["What kinds of cancer are there?", "List the types of cancer", "Tell me about breast cancer"],
    "cancer_symptoms": ["I have a lump and it won't go away", "I've been coughing blood recently", "What are warning signs of cancer?"],
    "cancer_prevention": ["How do I avoid cancer?", "Tips to reduce cancer risk", "Can cancer be prevented?"],

    "diabetes_general": ["Can you explain diabetes?", "What causes diabetes?", "I want to know about diabetes"],
    "diabetes_types": ["What kinds of diabetes exist?", "List diabetes types", "Tell me about type 1 vs type 2 diabetes"],
    "diabetes_symptoms": ["I'm always thirsty and tired", "What are signs of diabetes?", "I keep needing to pee a lot"],
    "diabetes_prevention": ["How do I avoid diabetes?", "Tips to reduce diabetes risk", "Can diabetes be prevented?"],

    "alzheimers_general": ["Can you explain Alzheimer's?", "What causes Alzheimer's disease?", "I want to know about Alzheimer's"],
    "alzheimers_types": ["What kinds of Alzheimer's are there?", "List Alzheimer's types", "Tell me about early onset Alzheimer's"],
    "alzheimers_symptoms": ["I keep forgetting things lately", "What are early signs of Alzheimer's?", "I'm getting confused about familiar places"],
    "alzheimers_prevention": ["How do I avoid Alzheimer's?", "Tips to reduce Alzheimer's risk", "Can Alzheimer's be prevented?"],

    "hypertension_general": ["Can you explain hypertension?", "What causes high blood pressure?", "I want to know about hypertension"],
    "hypertension_types": ["What kinds of hypertension exist?", "List hypertension types", "Tell me about primary vs secondary hypertension"],
    "hypertension_symptoms": ["My chest feels tight and I have a headache", "What are signs of high blood pressure?", "I feel short of breath and dizzy"],
    "hypertension_prevention": ["How do I avoid high blood pressure?", "Tips to reduce hypertension risk", "Can hypertension be prevented?"],

    "stroke_general": ["Can you explain a stroke?", "What causes a stroke?", "I want to know about strokes"],
    "stroke_types": ["What kinds of strokes are there?", "List stroke types", "Tell me about ischemic vs hemorrhagic stroke"],
    "stroke_symptoms": ["My face feels numb on one side", "What are warning signs of a stroke?", "I'm slurring my words suddenly"],
    "stroke_prevention": ["How do I avoid a stroke?", "Tips to reduce stroke risk", "Can strokes be prevented?"],
}

def main():
    shutil.copy(INPUT_FILE, BACKUP_FILE)
    print(f"Backed up original to {BACKUP_FILE}")

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    added = 0
    for intent in data["intents"]:
        tag = intent["tag"]
        if tag in EXTRA_PATTERNS:
            new_patterns = [p for p in EXTRA_PATTERNS[tag] if p not in intent["patterns"]]
            intent["patterns"].extend(new_patterns)
            added += len(new_patterns)

    with open(INPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Added {added} new patterns across {len(EXTRA_PATTERNS)} tags")
    print(f"Saved updated {INPUT_FILE}")

if __name__ == "__main__":
    main()
