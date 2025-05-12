from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import uvicorn
import os
import json
from dotenv import load_dotenv
from clarification_orchestrator import ClarificationOrchestrator
from kirke_pipeline import KirkeAIPipeline, FieldValidator
from huggingface_hub import InferenceClient
from langchain_core.prompts import ChatPromptTemplate
from huggingface_hub import AsyncInferenceClient
import re

class StrOutputParser:
    def invoke(self, response: str) -> str:
        return str(response)

# Load environment variables from .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# ---------------------------
# Load Schema
# ---------------------------
with open(os.path.join(os.path.dirname(__file__), "Schema_Loader.json"), "r") as f:
    kirke_schema = json.load(f)


# ---------------------------
# LLM-based Field Extractor
# ---------------------------
class LLMFieldExtractor:
    huggingface_token = os.getenv("HF_key")
    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.3"


    def __init__(self, schema: Dict[str, List[str]]):
        self.schema = schema
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")

        print("Initializing LLMFieldExtractor...")
        self.schema = schema

        self.llm_client = AsyncInferenceClient(
            api_key=self.huggingface_token,
            timeout=120
        )

        self.prompt = ChatPromptTemplate.from_template("""You are an information extraction system. Extract the required entities from the input statement based on the definitions below and return them in strict JSON format. Do not include any explanation, introduction, or extra text—only the final JSON.

Entity Definitions:

Activity_Type: A concise label for the primary network activity, such as "Upgrade", "Installation", "Maintenance", etc.
Network_Name: The name or description of the main network, system, or segment impacted (e.g., Core, Transport, RAN).
Service_Impact_Level: The estimated level of service impact due to the activity (e.g., High, Medium, Low, None, Service Affecting, Non Service Affecting).
Change_Type: The classification of the change (e.g., Normal, Expedite, Breakfix, Emergency).                                                     

If you are not able to accurately extract a field, return "NO" as the value for that field.

Return the result in JSON format with these keys:
{{
  "Activity_Type": "",
  "Network_Name": "",
  "Service_Impact_Level": "",
  "Change_Type": "",
}}

Statement: {question}
Extracted Entities:""")

        self.output_parser = StrOutputParser()

    @staticmethod
    def convert_model_output(model_output: str):
    # Try to load full string as JSON first
        try:
            parsed = json.loads(model_output)
            print("Successfully parsed full model output as JSON.")
        except json.JSONDecodeError:
            print("Direct JSON parse failed. Attempting regex extraction...")
            # Fallback: regex match for JSON object with list support
            json_block_matches = list(re.finditer(
                r'"Activity_Type":\s*"[^"]*",\s*'
                r'"Network_Name":\s*"[^"]*",\s*'
                r'"Service_Impact_Level":\s*"[^"]*",\s*'
                r'"Change_Type":\s*"[^"]*",\s*',
                model_output,
                re.DOTALL
            ))

            if not json_block_matches:
                print("No valid JSON block found using regex.")
                return None

            last_json_block = json_block_matches[-1].group(0)
            parsed = json.loads(last_json_block)
            print("Successfully parsed JSON from regex match.")

        # Normalize into expected structure
        return {
            "network_name": {"value": parsed.get("Network_Name")},
            "activity_type": {"value": parsed.get("Activity_Type")},
            "impact_level": {"value": parsed.get("Service_Impact_Level")},
            "change_type": {"value": parsed.get("Change_Type")}
        }
    
    async def extract(self, user_input: str) -> Dict[str, Dict[str, str]]:
        print(f"Extracting from input: {user_input}")
        formatted_prompt = self.prompt.format(question=user_input)
        print(formatted_prompt)
        #raw_output = await self.llm_client.text_generation(formatted_prompt) #check this line
        response = await self.llm_client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": formatted_prompt
            }],
            model="mistralai/Mistral-7B-Instruct-v0.3",
            temperature=0.3,
        )
        raw_output = response.choices[0].message.content
        final_output = self.output_parser.invoke(raw_output)
        print(f"Raw LLM output:\n{final_output}")
        final_output=str(final_output)
        extracted_fields = self.convert_model_output(final_output)
        print(f"Extracted fields:\n{extracted_fields}")
        if extracted_fields is None:
            raise ValueError(f"No valid JSON block found in model output. Got:\n{final_output}")
        return extracted_fields
    
        

# ---------------------------
# FastAPI Models
# ---------------------------
class CRInputRequest(BaseModel):
    user_text: str
    use_llm: Optional[bool] = False


# Model for field values (simpler format for input/output)
class FieldValues(BaseModel):
    network_name: Optional[str] = None
    activity_type: Optional[str] = None
    impact_level: Optional[str] = None
    change_type: Optional[str] = None


# Model for the process_input response
class ProcessResponse(BaseModel):
    values: FieldValues
    clarification_needs: Dict[str, Dict[str, Any]]
    completed: bool


# ---------------------------
# Init Services
# ---------------------------
openai_api_key = os.getenv("OPENAI_API_KEY")
pipeline = KirkeAIPipeline(schema=kirke_schema, openai_api_key=openai_api_key)
llm_extractor = LLMFieldExtractor(kirke_schema)
clarifier = ClarificationOrchestrator(kirke_schema)
validator = FieldValidator()
validator.load_schema(kirke_schema)



# Helper functions
def extract_values_from_fields(extracted_fields: Dict[str, Dict]) -> Dict[str, str]:
    """Extract simple field values from the complex extracted_fields structure"""
    values = {}
    for field, details in extracted_fields.items():
        if details.get("status") == "matched":
            values[field] = details.get("matched_value")
    return values



def create_clarification_needs(extracted_fields: Dict[str, Dict], prompts: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Create a structure showing what needs clarification"""
    needs = {}
    for field, prompt in prompts.items():
        field_details = extracted_fields.get(field, {})
        needs[field] = {
            "prompt": prompt,
            "suggested_options": field_details.get("suggested_matches", [])
            }
    return needs


# ---------------------------
# FastAPI Setup
# ---------------------------
app = FastAPI(
    title="Kirke AI - Change Plan Assistant",
    description="API to process change request text and extract/validate form fields using LLM or embeddings.",
    version="1.0"
)


@app.post("/process_input", response_model=ProcessResponse)
async def process_input(data: CRInputRequest):
    """
    Process user input text to extract fields.

    Returns:
        - values: Successfully matched values
        - clarification_needs: Fields needing clarification with prompts and options
        - completed: Flag indicating if all fields are complete
    """
    try:
        if data.use_llm:
            fields = await llm_extractor.extract(data.user_text)
            validated_fields = validator.validate_fields(fields)
            result = {"extracted_fields": validated_fields, "method": "llm"}
            print("*********************************")
            print(result)
        else:
            result = pipeline.process_input(data.user_text)

        # Generate clarification prompts
        prompts = clarifier.generate_chat_prompts(result["extracted_fields"])

        # Extract simple values for complete fields
        values = extract_values_from_fields(result["extracted_fields"])

        # Create clarification needs structure
        clarification_needs = create_clarification_needs(result["extracted_fields"], prompts)

        # Check if all fields are complete
        completed = len(prompts) == 0

        # Create a FieldValues object with extracted values
        field_values = FieldValues(**values)

        return {
            "values": field_values,
            "clarification_needs": clarification_needs,
            "completed": completed
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@app.post("/clarify_fields", response_model=ProcessResponse)
def clarify_fields(data: FieldValues):
    """
    Update fields with user-provided values.

    Takes:
        - Field values to update (only include fields that need updating)

    Returns:
        - Complete set of values (including previously matched ones)
        - Any remaining fields that need clarification
        - Completed flag
    """
    try:
        # Convert input to dictionary and filter out None values
        provided_values = {k: v for k, v in data.dict().items() if v is not None}

        # Validate the provided values
        validated_fields = {}
        for field, value in provided_values.items():
            # Check if the field exists in the schema
            if field not in kirke_schema:
                continue

            # Check if the value is in the allowed values
            allowed_values = kirke_schema[field]
            if value.lower() in [v.lower() for v in allowed_values]:
                validated_fields[field] = {
                    "extracted_value": value,
                    "matched_value": value,
                    "similarity": "1.0",
                    "status": "matched",
                    "suggested_matches": value
                }
            else:
                # Value not in allowed options, mark as ambiguous
                validated_fields[field] = {
                    "extracted_value": value,
                    "matched_value": "",
                    "similarity": "0.0",
                    "status": "ambiguous",
                    "suggested_matches": allowed_values[:3] if allowed_values else []
                }

        # Get all fields from schema
        all_fields = set(kirke_schema.keys())

        # Fields not provided are marked as missing
        for field in all_fields:
            if field not in validated_fields:
                validated_fields[field] = {
                    "extracted_value": "",
                    "matched_value": "",
                    "similarity": "0.0",
                    "status": "missing",
                    "suggested_matches": ",".join(kirke_schema[field][:3]) if kirke_schema[field] else ""
                }

        # Generate clarification prompts
        prompts = clarifier.generate_chat_prompts(validated_fields)

        # Extract simple values for matched fields
        values = extract_values_from_fields(validated_fields)

        # Create clarification needs structure
        clarification_needs = create_clarification_needs(validated_fields, prompts)

        # Check if all fields are complete
        completed = len(prompts) == 0

        # Create a FieldValues object with all known values
        field_values = FieldValues(**values)

        return {
            "values": field_values,
            "clarification_needs": clarification_needs,
            "completed": completed
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("kirke_api:app", host="127.0.0.1", port=8080, reload=True)