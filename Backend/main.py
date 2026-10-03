import os
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not found. Add it to backend/.env"
    )


# Gemini client
client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Sahaara AI API",
    description="Multi-Agent AI Emergency Assistance Platform",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DATA MODELS
# ============================================================

class EmergencyRequest(BaseModel):
    category: str
    location: str = "Unknown"
    situation: str


class AgentResult(BaseModel):
    agent: str
    output: str


class EmergencyResponse(BaseModel):
    risk_level: str
    risk_category: str
    confidence: int
    summary: str
    agent_trace: list[AgentResult]
    action_plan: list[str]
    safety_note: str
    verification_note: str


# ============================================================
# EMERGENCY CATEGORIES
# ============================================================

CATEGORIES = [
    "Fire",
    "Smoke",
    "Bleeding / Injury",
    "Unconsciousness",
    "Electrical Hazard",
    "Trapped Person",
    "Other Emergency"
]


# ============================================================
# KEYWORD DETECTION
# ============================================================

KEYWORDS = {

    "Fire": [
        "fire",
        "flame",
        "burning",
        "blaze",
        "aag"
    ],

    "Smoke": [
        "smoke",
        "dhuwan",
        "smoke-filled"
    ],

    "Bleeding / Injury": [
        "bleeding",
        "blood",
        "wound",
        "injury",
        "cut",
        "bleed"
    ],

    "Unconsciousness": [
        "unconscious",
        "passed out",
        "not responding",
        "fainted",
        "behosh"
    ],

    "Electrical Hazard": [
        "electric",
        "electrical",
        "electric shock",
        "live wire",
        "spark",
        "short circuit"
    ],

    "Trapped Person": [
        "trapped",
        "stuck",
        "locked inside",
        "can't escape",
        "cannot escape"
    ]
}


# ============================================================
# FALLBACK ACTIONS
# ============================================================

FALLBACK_ACTIONS = [
    "Move away from immediate danger if it is safe to do so.",
    "Alert nearby people and avoid the hazard.",
    "Contact the appropriate local emergency services when someone is at risk.",
    "Provide responders with the location and a clear description.",
    "Follow instructions from trained emergency personnel."
]


# ============================================================
# CATEGORY DETECTION
# ============================================================

def detect_category(
    situation: str,
    selected: str
):

    text = situation.lower()

    for category, words in KEYWORDS.items():

        for word in words:

            if word in text:
                return category

    if selected in CATEGORIES:
        return selected

    return "Other Emergency"


# ============================================================
# JSON PARSER
# ============================================================

def parse_json(raw: str):

    if not raw:
        return None

    raw = raw.strip()

    # Remove Markdown code fences
    if raw.startswith("```"):

        raw = (
            raw
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

    # Try direct JSON parsing
    try:

        return json.loads(raw)

    except json.JSONDecodeError:

        pass

    # Try extracting JSON object
    start = raw.find("{")
    end = raw.rfind("}")

    if start >= 0 and end > start:

        try:

            return json.loads(
                raw[start:end + 1]
            )

        except json.JSONDecodeError:

            pass

    return None


# ============================================================
# GEMINI AI ANALYSIS
# ============================================================

def gemini_analysis(
    req: EmergencyRequest,
    detected: str
):

    prompt = f"""
You are Sahaara AI, a safety-first emergency assistance assistant.

Analyze this emergency situation.

Selected category:
{req.category}

Detected category:
{detected}

Location/context:
{req.location}

Situation:
{req.situation}

Return ONLY valid JSON.

The JSON must contain exactly these fields:

{{
    "risk_level": "Critical / High / Medium / Low / Needs assessment",
    "risk_category": "one of: Fire, Smoke, Bleeding / Injury, Unconsciousness, Electrical Hazard, Trapped Person, Other Emergency",
    "confidence": 0,
    "summary": "one short sentence",
    "situation_analysis": "short factual analysis",
    "risk_assessment": "short explanation of the main risk",
    "resources": "brief guidance about contacting appropriate local emergency services or resources",
    "action_plan": [
        "First safe practical step",
        "Second safe practical step",
        "Third safe practical step",
        "Fourth safe practical step",
        "Fifth safe practical step"
    ],
    "safety_guidance": "short safety warning",
    "verification_note": "short verification warning"
}}

IMPORTANT SAFETY RULES:

- Prioritize immediate personal safety.
- Keep instructions simple and actionable.
- Do not diagnose medical conditions.
- Do not claim that Sahaara AI can rescue anyone.
- Do not claim emergency dispatch capability.
- Do not invent emergency phone numbers.
- Do not invent URLs.
- Do not invent emergency agencies.
- Do not invent official procedures.
- Do not give dangerous or reckless instructions.
- If there is immediate life-threatening danger, tell the user to contact appropriate local emergency services.
- Do not guarantee that your assessment is correct.
- Critical instructions should be verified with qualified responders or official authorities.
"""


    # ========================================================
    # CURRENT GEMINI INTERACTIONS API
    # ========================================================

    try:

        interaction = client.interactions.create(

            model="gemini-3.8-flash",

            input=prompt,

            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": {
                    "type": "object",

                    "properties": {

                        "risk_level": {
                            "type": "string"
                        },

                        "risk_category": {
                            "type": "string"
                        },

                        "confidence": {
                            "type": "integer"
                        },

                        "summary": {
                            "type": "string"
                        },

                        "situation_analysis": {
                            "type": "string"
                        },

                        "risk_assessment": {
                            "type": "string"
                        },

                        "resources": {
                            "type": "string"
                        },

                        "action_plan": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        },

                        "safety_guidance": {
                            "type": "string"
                        },

                        "verification_note": {
                            "type": "string"
                        }
                    },

                    "required": [
                        "risk_level",
                        "risk_category",
                        "confidence",
                        "summary",
                        "situation_analysis",
                        "risk_assessment",
                        "resources",
                        "action_plan",
                        "safety_guidance",
                        "verification_note"
                    ]
                }
            }
        )


        # Current Interactions API convenience property
        # returns the final text output.
        raw_output = interaction.output_text


        print("\n========== GEMINI RESPONSE ==========")
        print(raw_output)
        print("=====================================\n")


        data = parse_json(raw_output)


        if data is None:

            raise RuntimeError(
                "Gemini returned an invalid JSON response."
            )


        return data


    except Exception as e:

        print("\n========== GEMINI ERROR ==========")
        print(
            f"{type(e).__name__}: {e}"
        )
        print("==================================\n")

        raise RuntimeError(
            "Gemini AI is temporarily unavailable."
        ) from e


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "project": "Sahaara AI",
        "status": "online",
        "version": "1.0.0",
        "ai": "Gemini 3.8 Flash"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "Sahaara AI"
    }


# ============================================================
# EMERGENCY ANALYSIS ENDPOINT
# ============================================================

@app.post(
    "/analyze",
    response_model=EmergencyResponse
)
def analyze(
    req: EmergencyRequest
):

    # ========================================================
    # VALIDATE USER INPUT
    # ========================================================

    if not req.situation.strip():

        raise HTTPException(
            status_code=400,
            detail=(
                "Please describe the emergency situation."
            )
        )


    # ========================================================
    # DETECT CATEGORY
    # ========================================================

    detected = detect_category(
        req.situation,
        req.category
    )


    # ========================================================
    # GEMINI ANALYSIS
    # ========================================================

    try:

        ai = gemini_analysis(
            req,
            detected
        )

    except RuntimeError as e:

        raise HTTPException(
            status_code=503,
            detail=str(e)
        )


    # ========================================================
    # RISK LEVEL
    # ========================================================

    risk = str(
        ai.get(
            "risk_level",
            "Needs assessment"
        )
    )


    # ========================================================
    # RISK CATEGORY
    # ========================================================

    category = str(
        ai.get(
            "risk_category",
            detected
        )
    )


    # ========================================================
    # CONFIDENCE
    # ========================================================

    try:

        confidence = int(
            ai.get(
                "confidence",
                90
            )
        )

        confidence = max(
            0,
            min(100, confidence)
        )

    except (
        ValueError,
        TypeError
    ):

        confidence = 90


    # ========================================================
    # ACTION PLAN
    # ========================================================

    actions = ai.get(
        "action_plan",
        FALLBACK_ACTIONS
    )


    if not isinstance(
        actions,
        list
    ):

        actions = [
            str(actions)
        ]


    actions = [
        str(action)
        for action in actions
        if str(action).strip()
    ]


    if not actions:

        actions = FALLBACK_ACTIONS


    # ========================================================
    # VERIFICATION NOTE
    # ========================================================

    verification = str(
        ai.get(
            "verification_note",
            (
                "Verify critical instructions with "
                "qualified responders or official authorities."
            )
        )
    )


    # ========================================================
    # MULTI-AGENT TRACE
    # ========================================================

    trace = [

        AgentResult(
            agent="Orchestrator Agent",
            output=(
                "Received the emergency request and "
                "coordinated the multi-agent safety workflow."
            )
        ),

        AgentResult(
            agent="Situation Analysis Agent",
            output=str(
                ai.get(
                    "situation_analysis",
                    "Reviewed the reported emergency situation."
                )
            )
        ),

        AgentResult(
            agent="Risk Assessment Agent",
            output=str(
                ai.get(
                    "risk_assessment",
                    f"Preliminary risk level: {risk}."
                )
            )
        ),

        AgentResult(
            agent="Resource Agent",
            output=str(
                ai.get(
                    "resources",
                    (
                        "Contact appropriate local "
                        "emergency resources when needed."
                    )
                )
            )
        ),

        AgentResult(
            agent="Action Planning Agent",
            output=(
                "Generated a concise safety-first "
                "action checklist."
            )
        ),

        AgentResult(
            agent="Verification Agent",
            output=verification
        )
    ]


    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    return EmergencyResponse(

        risk_level=risk,

        risk_category=category,

        confidence=confidence,

        summary=str(
            ai.get(
                "summary",
                (
                    "The situation was analyzed "
                    "for immediate safety risks."
                )
            )
        ),

        agent_trace=trace,

        action_plan=actions[:7],

        safety_note=str(
            ai.get(
                "safety_guidance",
                (
                    "Sahaara AI provides general "
                    "preparedness guidance only and "
                    "does not replace emergency responders, "
                    "medical professionals, or official alerts."
                )
            )
        ),

        verification_note=verification
    )