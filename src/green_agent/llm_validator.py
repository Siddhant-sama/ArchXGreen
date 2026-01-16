"""
LLM-Based Architectural Validator

Uses LLM reasoning to validate complex architectural constraints that are
difficult to check with pattern matching:
- Hierarchical structure
- Algorithm-specific patterns
- Sequential/pipelined design
- Parameter usage
- Module instantiation patterns
"""

import os
import json
from typing import Dict, List, Optional
from dataclasses import dataclass

from .models import ArchitecturalCompliance


@dataclass
class LLMValidationResult:
    """Result from LLM validation"""
    is_compliant: bool
    confidence: float  # 0.0 to 1.0
    violations: List[str]
    warnings: List[str]
    strengths: List[str]
    reasoning: str


class LLMArchitecturalValidator:
    """Uses LLM to validate complex architectural requirements"""
    
    VALIDATION_PROMPT_TEMPLATE = """You are an expert hardware design validator. Your task is to evaluate whether a Verilog implementation complies with the given design specifications and constraints.

**Problem Description:**
{problem_description}

**Design Specifications:**
{design_specs}

**Submitted Verilog Code:**
```verilog
{verilog_code}
```

**Your Task:**
Carefully analyze the submitted code and determine if it follows ALL requirements from the problem description and design specifications. Focus on:

1. **Hierarchical Structure**: If the spec requires hierarchical design (e.g., "using two 4-bit modules"), check if sub-modules are properly instantiated
2. **Algorithm Patterns**: If specific algorithms are required (e.g., "Kogge-Stone prefix tree", "Dadda multiplier strategy"), verify the implementation follows that algorithm
3. **Sequential vs Combinational**: Check if the design properly uses sequential logic (registers, FSMs) when required, or is purely combinational when specified
4. **Pipeline Stages**: If pipelining is required, verify the correct number of pipeline stages and register placement
5. **Parameters**: If the design should be parameterizable, check parameter usage and constraints
6. **Module Instantiations**: Verify that required sub-modules are instantiated correctly
7. **Special Requirements**: Any other specific constraints mentioned in the specs

**Response Format (JSON):**
```json
{{
  "is_compliant": true/false,
  "confidence": 0.0-1.0,
  "violations": ["List of requirement violations found"],
  "warnings": ["List of potential issues or concerns"],
  "strengths": ["List of things done well"],
  "reasoning": "Detailed explanation of your evaluation"
}}
```

**Important:**
- Be strict about explicit requirements
- Be lenient about implementation details not specified
- Focus on architectural correctness, not code style
- If a requirement is ambiguous, mention it in warnings
- Provide specific references to lines/constructs in the code

Evaluate the code now:"""

    def __init__(self, llm_provider: str = "openai", model: str = "gpt-4", api_key: Optional[str] = None):
        """
        Initialize LLM validator
        
        Args:
            llm_provider: "openai", "anthropic", or "mock" (for testing)
            model: Model name (e.g., "gpt-4", "claude-3-opus")
            api_key: API key (or set via environment variable)
        """
        self.llm_provider = llm_provider.lower()
        self.model = model
        self.api_key = api_key or self._get_api_key()
        
        # Initialize LLM client based on provider
        if self.llm_provider == "openai":
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key) if self.api_key else None
            except ImportError:
                self.client = None
                print("Warning: openai package not installed. LLM validation disabled.")
        elif self.llm_provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else None
            except ImportError:
                self.client = None
                print("Warning: anthropic package not installed. LLM validation disabled.")
        elif self.llm_provider == "mock":
            self.client = "mock"  # For testing without API calls
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment"""
        if self.llm_provider == "openai":
            return os.getenv("OPENAI_API_KEY")
        elif self.llm_provider == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY")
        return None
    
    def is_available(self) -> bool:
        """Check if LLM validation is available"""
        return self.client is not None
    
    def validate(
        self,
        verilog_code: str,
        problem_description: str,
        design_specs: str
    ) -> Optional[LLMValidationResult]:
        """
        Validate code using LLM reasoning
        
        Returns:
            LLMValidationResult or None if LLM unavailable
        """
        if not self.is_available():
            return None
        
        # Generate prompt
        prompt = self.VALIDATION_PROMPT_TEMPLATE.format(
            problem_description=problem_description,
            design_specs=design_specs,
            verilog_code=verilog_code
        )
        
        # Call LLM
        try:
            if self.llm_provider == "openai":
                response = self._call_openai(prompt)
            elif self.llm_provider == "anthropic":
                response = self._call_anthropic(prompt)
            elif self.llm_provider == "mock":
                response = self._call_mock(prompt)
            else:
                return None
            
            # Parse response
            result = self._parse_llm_response(response)
            return result
            
        except Exception as e:
            print(f"Warning: LLM validation failed: {str(e)}")
            return None
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert hardware design validator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for consistent validation
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    
    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            temperature=0.1,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    
    def _call_mock(self, prompt: str) -> str:
        """Mock LLM for testing"""
        # Return a mock compliant response
        return json.dumps({
            "is_compliant": True,
            "confidence": 0.95,
            "violations": [],
            "warnings": ["Mock validation - LLM not actually called"],
            "strengths": ["Code appears structurally sound (mock evaluation)"],
            "reasoning": "This is a mock validation response for testing. No actual LLM was called."
        })
    
    def _parse_llm_response(self, response: str) -> LLMValidationResult:
        """Parse LLM JSON response into result object"""
        try:
            # Try to extract JSON from response
            # Handle cases where LLM includes markdown code blocks
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                response = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                response = response[start:end].strip()
            
            data = json.loads(response)
            
            return LLMValidationResult(
                is_compliant=data.get("is_compliant", False),
                confidence=data.get("confidence", 0.5),
                violations=data.get("violations", []),
                warnings=data.get("warnings", []),
                strengths=data.get("strengths", []),
                reasoning=data.get("reasoning", "")
            )
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse LLM response as JSON: {e}")
            print(f"Response: {response[:200]}...")
            # Return a failed validation
            return LLMValidationResult(
                is_compliant=False,
                confidence=0.0,
                violations=["Failed to parse LLM validation response"],
                warnings=[],
                strengths=[],
                reasoning=f"Error parsing LLM response: {str(e)}"
            )


def merge_validation_results(
    rule_based: ArchitecturalCompliance,
    llm_based: Optional[LLMValidationResult]
) -> ArchitecturalCompliance:
    """
    Merge rule-based and LLM-based validation results
    
    Strategy:
    - If LLM is unavailable, return rule-based only
    - If both available, combine violations and adjust score
    - Weight: 60% rule-based (objective), 40% LLM-based (subjective)
    """
    if llm_based is None:
        # No LLM validation - return rule-based only
        return rule_based
    
    # Create merged compliance
    merged = ArchitecturalCompliance()
    
    # Merge violations and warnings
    merged.violations = rule_based.violations + llm_based.violations
    merged.warnings = rule_based.warnings + llm_based.warnings
    
    # Update compliance status
    merged.is_compliant = rule_based.is_compliant and llm_based.is_compliant
    
    # Merge compliance flags
    merged.structural_compliance = rule_based.structural_compliance
    merged.interface_compliance = rule_based.interface_compliance
    merged.hierarchy_compliance = llm_based.is_compliant  # LLM checks hierarchy
    merged.constraint_compliance = rule_based.constraint_compliance and llm_based.is_compliant
    
    # Calculate weighted score
    # 60% rule-based (objective), 40% LLM (subjective)
    rule_weight = 0.6
    llm_weight = 0.4
    merged.compliance_score = (
        rule_based.compliance_score * rule_weight +
        (llm_based.confidence if llm_based.is_compliant else 0.0) * llm_weight
    )
    
    # Merge metadata
    merged.required_style = rule_based.required_style
    merged.detected_style = rule_based.detected_style
    merged.required_constraints = rule_based.required_constraints
    
    # Generate merged validation log
    merged.validation_log = _generate_merged_log(rule_based, llm_based)
    
    return merged


def _generate_merged_log(
    rule_based: ArchitecturalCompliance,
    llm_based: LLMValidationResult
) -> str:
    """Generate combined validation log"""
    log_lines = ["Architectural Validation Report (Hybrid)", "=" * 50]
    
    log_lines.append("\n📐 Rule-Based Validation:")
    log_lines.append(f"   Score: {rule_based.compliance_score:.2f}")
    log_lines.append(f"   Style: {rule_based.detected_style or 'Unknown'}")
    if rule_based.violations:
        log_lines.append(f"   Violations: {len(rule_based.violations)}")
    
    log_lines.append("\n🤖 LLM-Based Validation:")
    log_lines.append(f"   Compliant: {'YES' if llm_based.is_compliant else 'NO'}")
    log_lines.append(f"   Confidence: {llm_based.confidence:.2f}")
    if llm_based.violations:
        log_lines.append(f"   Violations: {len(llm_based.violations)}")
    
    if llm_based.strengths:
        log_lines.append(f"\n✅ Strengths:")
        for strength in llm_based.strengths:
            log_lines.append(f"   • {strength}")
    
    if rule_based.violations or llm_based.violations:
        log_lines.append(f"\n❌ All Violations:")
        for v in rule_based.violations + llm_based.violations:
            log_lines.append(f"   • {v}")
    
    if rule_based.warnings or llm_based.warnings:
        log_lines.append(f"\n⚠️  All Warnings:")
        for w in rule_based.warnings + llm_based.warnings:
            log_lines.append(f"   • {w}")
    
    if llm_based.reasoning:
        log_lines.append(f"\n💭 LLM Reasoning:")
        log_lines.append(f"   {llm_based.reasoning[:300]}...")
    
    return '\n'.join(log_lines)


# Singleton instance
_llm_validator: Optional[LLMArchitecturalValidator] = None


def get_llm_validator(
    llm_provider: str = "openai",
    model: str = "gpt-4",
    enable: bool = True
) -> Optional[LLMArchitecturalValidator]:
    """
    Get or create LLM validator instance
    
    Args:
        llm_provider: "openai", "anthropic", or "mock"
        model: Model name
        enable: Whether to enable LLM validation
    """
    global _llm_validator
    
    if not enable:
        return None
    
    if _llm_validator is None:
        try:
            _llm_validator = LLMArchitecturalValidator(
                llm_provider=llm_provider,
                model=model
            )
        except Exception as e:
            print(f"Warning: Failed to initialize LLM validator: {e}")
            return None
    
    return _llm_validator
