import re
import os
import json
import numpy as np
from typing import Dict, List, Any, Optional

from sentence_transformers import SentenceTransformer, util
import faiss
from langchain_core.prompts import ChatPromptTemplate
from huggingface_hub import AsyncInferenceClient

# from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()


class StrOutputParser:
    def invoke(self, response: str) -> str:
        return str(response)

# ------------------------
# Text Preprocessing
# ------------------------
class TextPreprocessor:
    def __init__(self, abbreviations: Optional[Dict[str, str]] = None):
        self.abbreviations = {
            "n/w": "network",
            "svc": "service",
            "int": "interface",
            "cat": "category"
        }
        if abbreviations:
            self.abbreviations.update(abbreviations)

    def expand_abbreviations(self, text: str) -> str:
        for abbr, full in self.abbreviations.items():
            text = re.sub(rf'\b{re.escape(abbr)}\b', full, text, flags=re.IGNORECASE)
        return text

    def clean_text(self, text: str, to_lowercase: bool = False) -> str:
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9\s.,-]', '', text)
        text = re.sub(r'[-_]{2,}', ' ', text)
        text = re.sub(r'[\t\r\f\v]', '', text)
        if to_lowercase:
            text = text.lower()
        return self.expand_abbreviations(text)

    def preprocess(self, text: str, to_lowercase: bool = False) -> str:
        return self.clean_text(text, to_lowercase)

# ------------------------
# Field Extraction
# ------------------------
class FieldExtractor:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.schema = {}
        self.field_embeddings = {}

    def load_schema(self, schema_json: Dict[str, List[str]]):
        self.schema = schema_json
        self.field_embeddings = {
            field: self.model.encode(values, convert_to_tensor=True)
            for field, values in schema_json.items()
        }

    def extract_fields(self, user_text: str) -> Dict[str, Dict[str, Any]]:
        if not self.field_embeddings:
            raise ValueError("Schema not loaded. Call load_schema() first.")

        input_embedding = self.model.encode(user_text, convert_to_tensor=True)
        results = {}

        for field, value_embeddings in self.field_embeddings.items():
            cosine_scores = util.cos_sim(input_embedding, value_embeddings)[0]
            best_idx = int(np.argmax(cosine_scores))
            best_score = float(cosine_scores[best_idx])
            best_match = self.schema[field][best_idx]

            results[field] = {
                "value": best_match,
                "confidence": round(best_score, 4)
            }

        return results

# ------------------------
# Field Validation
# ------------------------
class FieldValidator:
    def __init__(self, model_name='all-MiniLM-L6-v2', similarity_threshold=0.9, ambiguity_threshold=0.7, top_k=3):
        """
        Initialize the field validator with embedding model and similarity thresholds.
        """
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.ambiguity_threshold = ambiguity_threshold
        self.top_k = top_k
        self.schema = {}
        self.faiss_indexes = {}

    def load_schema(self, schema_json: dict):
        """
        Build FAISS index for each schema field.
        """
        self.schema = schema_json
        dim = self.model.get_sentence_embedding_dimension()

        for field, values in schema_json.items():
            embeddings = self.model.encode(values)
            faiss.normalize_L2(embeddings)  # required for cosine similarity
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)
            self.faiss_indexes[field] = (index, embeddings, values)

    def validate_field(self, field_name: str, extracted_value: str):
        """
        Validate a single field's extracted value using semantic similarity.
        If the extracted value is "NO", mark it as missing immediately.
        """
        if extracted_value.strip().upper() == "NO":
            return {
                "extracted_value": extracted_value,
                "matched_value": None,
                "similarity": 0.0,
                "status": "missing",
                "suggested_matches": []
            }

        #For fields like Location, Date, Network Elements
        if field_name not in self.schema and extracted_value:
            return {
                "extracted_value": extracted_value,
                "matched_value": None,
                "similarity": 0.0,
                "status": "Extracted from LLM",
                "suggested_matches": []
            }



        if field_name in self.schema:

            index, embeddings, values = self.faiss_indexes[field_name]

            # Encode and normalize input
            embedding = self.model.encode([extracted_value])
            faiss.normalize_L2(embedding)

            # Search top-K matches
            D, I = index.search(embedding, self.top_k)

            best_score = float(D[0][0])
            best_match = values[I[0][0]]
            suggested_matches = [values[i] for i in I[0]]

            # Determine validation status
            if best_score >= self.similarity_threshold:
                status = "matched"
            elif best_score < self.ambiguity_threshold:
                status = "ambiguous"

            return {
                "extracted_value": extracted_value,
                "matched_value": best_match,
                "similarity": round(best_score, 4),
                "status": status,
                "suggested_matches": suggested_matches
            }

    def validate_fields(self, extracted_fields: dict):
        """
        Validate only the fields from extracted_fields that are present in the schema.
        """
        validated = {}
        for field_name, info in extracted_fields.items():
            # Only validate fields that exist in the schema
                result = self.validate_field(field_name, info["value"])
                validated[field_name] = result
        return validated
    
# ------------------------
# Clarification Generator
# ------------------------
class ClarificationGenerator:
    huggingface_token = os.getenv("HF_key")
    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    def __init__(self, api_key: str = None, model_name: str = "gpt-3.5-turbo"):
        self.llm_client = AsyncInferenceClient(
            model=self.repo_id,
            api_key=self.huggingface_token,
            timeout=120
        )
        self.prompt = ChatPromptTemplate.from_template(""""We parsed the user input for the field '{field}' and inferred it might refer to one of the following: {choices}. "
            "The input text was: '{value}'. Please generate a concise and polite question asking the user to confirm the correct option "
            "or provide a new one. Avoid implying that the user's input was incorrect.""")

        self.output_parser = StrOutputParser()

    def generate_prompt(self, field: str, value: str, choices: List[str]) -> str:
        # return self.chain.invoke({"field": field, "value": value, "choices":", ".join(choices))
        return self.chain.invoke({"field": field, "value": value, "choices": choices})
    def generate_for_all(self, ambiguous_fields: Dict[str, Dict]) -> Dict[str, str]:
        prompts = {}
        for field, details in ambiguous_fields.items():
            choices = details.get("suggested_matches", [details.get("matched_value", "")])
            prompts[field] = self.generate_prompt(field, details["extracted_value"], choices)
        return prompts

class ClarificationResponseHandler:
    def __init__(self, field_validator):
        self.field_validator = field_validator

    def apply_corrections(self, extracted_fields: Dict[str, Dict], user_corrections: Dict[str, str]) -> Dict[str, Dict]:
        updated_fields = extracted_fields.copy()
        for field_name, corrected_value in user_corrections.items():
            updated_fields[field_name] = {"value": corrected_value}
        return updated_fields

    def process_corrections(self, extracted_fields: Dict[str, Dict], user_corrections: Dict[str, str]) -> Dict[str, Dict]:
        updated_fields = self.apply_corrections(extracted_fields, user_corrections)
        return self.field_validator.validate_fields(updated_fields)

# ------------------------
# Main Pipeline
# ------------------------
class KirkeAIPipeline:
    def __init__(self, schema: Dict[str, List[str]], model_name: str = 'all-MiniLM-L6-v2', similarity_threshold: float = 0.95, ambiguity_threshold: float = 0.85, openai_api_key: Optional[str] = None):
        self.preprocessor = TextPreprocessor()
        self.extractor = FieldExtractor(model_name=model_name)
        self.validator = FieldValidator(model_name=model_name, similarity_threshold=similarity_threshold, ambiguity_threshold=ambiguity_threshold)

        self.has_llm = bool(openai_api_key or os.getenv('OPENAI_API_KEY'))
        if self.has_llm:
            self.clarifier = ClarificationGenerator(api_key=openai_api_key)
            self.correction_handler = ClarificationResponseHandler(self.validator)

        self.schema = schema
        self.extractor.load_schema(schema)
        self.validator.load_schema(schema)

    def process_input(self, user_text: str) -> Dict[str, Any]:
        processed_text = self.preprocessor.preprocess(user_text)
        extracted_fields = self.extractor.extract_fields(processed_text)
        validated_fields = self.validator.validate_fields(extracted_fields)

        missing_fields, ambiguous_fields, matched_fields = [], {}, {}

        for field, details in validated_fields.items():
            if details["status"] == "missing":
                missing_fields.append(field)
            elif details["status"] == "ambiguous":
                ambiguous_fields[field] = details
            else:
                matched_fields[field] = details

        clarification_prompts = {}
        if ambiguous_fields and self.has_llm:
            clarification_prompts = self.clarifier.generate_for_all(ambiguous_fields)

        return {
            "processed_text": processed_text,
            "extracted_fields": validated_fields,
            "missing_fields": missing_fields,
            "ambiguous_fields": ambiguous_fields,
            "matched_fields": matched_fields,
            "clarification_prompts": clarification_prompts
        }

    def handle_clarifications(self, extracted_fields: Dict[str, Dict], user_corrections: Dict[str, str]) -> Dict[str, Any]:
        if not self.has_llm:
            raise ValueError("LLM components not initialized. API key required.")

        revalidated_fields = self.correction_handler.process_corrections(extracted_fields, user_corrections)

        missing_fields, ambiguous_fields, matched_fields = [], {}, {}
        for field, details in revalidated_fields.items():
            if details["status"] == "missing":
                missing_fields.append(field)
            elif details["status"] == "ambiguous":
                ambiguous_fields[field] = details
            else:
                matched_fields[field] = details

        clarification_prompts = {}
        if ambiguous_fields:
            clarification_prompts = self.clarifier.generate_for_all(ambiguous_fields)

        return {
            "extracted_fields": revalidated_fields,
            "missing_fields": missing_fields,
            "ambiguous_fields": ambiguous_fields,
            "matched_fields": matched_fields,
            "clarification_prompts": clarification_prompts
        }
