import json
from typing import Dict, List, Tuple

class ClarificationOrchestrator:
    def __init__(self, schema_dict: Dict[str, List[str]]):
        self.schema_dict = schema_dict
        self.schema_fields = list(self.schema_dict.keys())
        print(f"[Init] Loaded schema fields: {self.schema_fields}")

    def identify_issues(self, validated_fields: Dict[str, Dict]) -> Tuple[List[str], List[str]]:
        print("\n[identify_issues] Validated fields received:")
        for field, val in validated_fields.items():
            print(f"  {field}: {val}")

        ambiguous = [
            field for field, val in validated_fields.items()
            if isinstance(val, dict) and val.get("status") == 'ambiguous'
        ]
        missing = [
            field for field, val in validated_fields.items()
            if isinstance(val, dict) and val.get("status") == 'missing'
        ]

        print(f"[identify_issues] Ambiguous fields: {ambiguous}")
        print(f"[identify_issues] Missing fields: {missing}")

        return ambiguous, missing

    def generate_prompts(self, ambiguous: List[str], missing: List[str], validated_fields: Dict[str, Dict]) -> Dict[str, str]:
        prompts = {}

        print("\n[generate_prompts] Generating prompts for ambiguous fields...")
        for field in ambiguous:
            val = validated_fields.get(field, {})
            suggestions = val.get("suggested_matches", [val.get("matched_value", "")])
            print(f"  Field: {field}, Suggestions: {suggestions}")
            prompt = f"The value for **{field}** seems unclear. Did you mean one of these: {', '.join(suggestions)}?"
            prompts[field] = prompt

        print("\n[generate_prompts] Generating prompts for missing fields...")
        for field in missing:
            options = self.schema_dict.get(field, [])
            print(f"  Field: {field}, Options: {options}")
            prompt = f"Could you please specify the **{field}**? Available options include: {', '.join(options)}."
            prompts[field] = prompt

        print(f"\n[generate_prompts] Final prompts generated:\n{json.dumps(prompts, indent=2)}")
        return prompts

    def generate_chat_prompts(self, validated_fields: Dict[str, Dict]) -> Dict[str, str]:
        print("\n[generate_chat_prompts] Starting clarification prompt generation...")
        ambiguous, missing = self.identify_issues(validated_fields)
        prompts = self.generate_prompts(ambiguous, missing, validated_fields)
        print("[generate_chat_prompts] Completed.")
        return prompts
