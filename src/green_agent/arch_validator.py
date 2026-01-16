"""
Architectural Validator for Verilog Code

Validates that submitted Verilog code complies with:
- Problem description constraints
- Design specification requirements
- Structural/architectural requirements (gate-level vs behavioral, etc.)
"""

import re
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path

from .models import ArchitecturalCompliance


class ArchitecturalValidator:
    """Validates Verilog code against design specifications and constraints"""
    
    # Keywords indicating behavioral constructs
    BEHAVIORAL_KEYWORDS = {
        'always', 'initial', 'if', 'else', 'case', 'casex', 'casez',
        'for', 'while', 'repeat', 'forever', '@', '#', 'posedge', 'negedge'
    }
    
    # Keywords indicating gate-level constructs
    GATE_LEVEL_KEYWORDS = {
        'and', 'or', 'not', 'nand', 'nor', 'xor', 'xnor',
        'buf', 'bufif0', 'bufif1', 'notif0', 'notif1'
    }
    
    def __init__(self):
        pass
    
    def validate(
        self,
        verilog_code: str,
        problem_description: str,
        design_specs: str,
        module_name: Optional[str] = None
    ) -> ArchitecturalCompliance:
        """
        Validate Verilog code against design constraints.
        
        Args:
            verilog_code: The submitted Verilog code
            problem_description: Problem description text
            design_specs: Design specifications text
            module_name: Expected module name (extracted from design specs if not provided)
        
        Returns:
            ArchitecturalCompliance object with validation results
        """
        compliance = ArchitecturalCompliance()
        
        # Parse requirements from specifications
        constraints = self._parse_constraints(problem_description)
        specs = self._parse_design_specs(design_specs)
        
        # Store requirements
        compliance.required_constraints = constraints.get('explicit_constraints', [])
        
        # Extract module info from submitted code
        submitted_module = self._extract_module_info(verilog_code)
        
        if not submitted_module:
            compliance.add_violation("No valid module declaration found in submitted code")
            return compliance
        
        # 1. Validate module interface (inputs/outputs)
        expected_module_name = module_name or specs.get('module_name')
        if expected_module_name:
            self._validate_interface(
                submitted_module,
                expected_module_name,
                specs.get('inputs', []),
                specs.get('outputs', []),
                compliance
            )
        
        # 2. Validate structural requirements (gate-level vs behavioral)
        required_style = self._determine_required_style(constraints)
        if required_style:
            compliance.required_style = required_style
            detected_style = self._detect_implementation_style(verilog_code)
            compliance.detected_style = detected_style
            
            self._validate_structural_style(
                verilog_code,
                required_style,
                detected_style,
                compliance
            )
        
        # 3. Validate specific design constraints
        self._validate_design_constraints(
            verilog_code,
            constraints,
            compliance
        )
        
        # 4. Check for forbidden constructs
        forbidden = constraints.get('forbidden', [])
        if forbidden:
            self._check_forbidden_constructs(verilog_code, forbidden, compliance)
        
        # Generate validation log
        compliance.validation_log = self._generate_validation_log(
            compliance, constraints, specs
        )
        
        return compliance
    
    def _parse_constraints(self, problem_description: str) -> Dict[str, any]:
        """Extract constraints from problem description"""
        constraints = {
            'explicit_constraints': [],
            'forbidden': [],
            'style_hints': []
        }
        
        lines = problem_description.lower().split('\n')
        
        for line in lines:
            # Look for constraint keywords
            if 'must' in line or 'constraint' in line or 'requirement' in line:
                constraints['explicit_constraints'].append(line.strip())
            
            # Detect style requirements
            if 'gate-level' in line or 'basic gates' in line:
                constraints['style_hints'].append('gate-level')
            elif 'behavioral' in line:
                constraints['style_hints'].append('behavioral')
            elif 'structural' in line or 'hierarchical' in line:
                constraints['style_hints'].append('structural')
            
            # Detect forbidden constructs
            if 'avoid' in line or 'no ' in line or 'without' in line:
                if 'behavioral' in line:
                    constraints['forbidden'].append('behavioral')
                if 'latch' in line:
                    constraints['forbidden'].append('latch')
        
        return constraints
    
    def _parse_design_specs(self, design_specs: str) -> Dict[str, any]:
        """Extract design specifications"""
        specs = {
            'module_name': None,
            'inputs': [],
            'outputs': [],
            'design_notes': []
        }
        
        # Extract module name
        module_match = re.search(r'module\s+(\w+)', design_specs)
        if module_match:
            specs['module_name'] = module_match.group(1)
        
        # Extract I/O specifications
        io_section = False
        for line in design_specs.split('\n'):
            line = line.strip()
            
            if line.startswith('Inputs:'):
                io_section = 'inputs'
                continue
            elif line.startswith('Outputs:'):
                io_section = 'outputs'
                continue
            elif line.startswith('Design Notes:') or line.startswith('Design Signature:'):
                io_section = False
                continue
            
            if io_section and line.startswith('-'):
                # Parse I/O line: "- signal_name [width] // comment"
                signal_match = re.match(r'-\s*(\w+)', line)
                if signal_match:
                    signal_name = signal_match.group(1)
                    if io_section == 'inputs':
                        specs['inputs'].append(signal_name)
                    elif io_section == 'outputs':
                        specs['outputs'].append(signal_name)
            
            # Extract design notes
            if 'design notes:' in line.lower():
                io_section = 'notes'
            elif io_section == 'notes' and line:
                specs['design_notes'].append(line)
        
        return specs
    
    def _extract_module_info(self, verilog_code: str) -> Optional[Dict[str, any]]:
        """Extract module declaration and I/O from Verilog code"""
        # Find module declaration
        module_pattern = r'module\s+(\w+)\s*(?:#\([^)]*\))?\s*\((.*?)\);'
        match = re.search(module_pattern, verilog_code, re.DOTALL)
        
        if not match:
            return None
        
        module_name = match.group(1)
        ports_text = match.group(2)
        
        # Parse port list
        inputs = []
        outputs = []
        inouts = []
        
        # Remove comments and clean up
        ports_text = re.sub(r'//.*', '', ports_text)
        ports_text = re.sub(r'/\*.*?\*/', '', ports_text, flags=re.DOTALL)
        
        # Parse port declarations
        port_lines = ports_text.split(',')
        for port_line in port_lines:
            port_line = port_line.strip()
            if not port_line:
                continue
            
            # Match: input/output [width] name
            port_match = re.search(r'(input|output|inout)\s+(?:\[[^\]]+\]\s*)?(\w+)', port_line)
            if port_match:
                direction = port_match.group(1)
                name = port_match.group(2)
                
                if direction == 'input':
                    inputs.append(name)
                elif direction == 'output':
                    outputs.append(name)
                elif direction == 'inout':
                    inouts.append(name)
        
        return {
            'name': module_name,
            'inputs': inputs,
            'outputs': outputs,
            'inouts': inouts
        }
    
    def _determine_required_style(self, constraints: Dict) -> Optional[str]:
        """Determine required implementation style from constraints"""
        style_hints = constraints.get('style_hints', [])
        
        if 'gate-level' in style_hints:
            return 'gate-level'
        elif 'behavioral' in style_hints:
            return 'behavioral'
        elif 'structural' in style_hints:
            return 'structural'
        
        # Check explicit constraints
        explicit = ' '.join(constraints.get('explicit_constraints', [])).lower()
        if 'gate-level' in explicit or 'basic gates' in explicit:
            return 'gate-level'
        elif 'behavioral' in explicit:
            return 'behavioral'
        
        return None
    
    def _detect_implementation_style(self, verilog_code: str) -> str:
        """Detect the implementation style of submitted code"""
        # Remove comments and strings
        code_clean = re.sub(r'//.*', '', verilog_code)
        code_clean = re.sub(r'/\*.*?\*/', '', code_clean, flags=re.DOTALL)
        code_clean = re.sub(r'"[^"]*"', '', code_clean)
        
        # Tokenize
        tokens = re.findall(r'\b\w+\b', code_clean.lower())
        token_set = set(tokens)
        
        # Check for behavioral keywords
        behavioral_count = len(token_set & self.BEHAVIORAL_KEYWORDS)
        gate_level_count = len(token_set & self.GATE_LEVEL_KEYWORDS)
        
        # Check for assign statements (continuous assignment - can be gate-level or behavioral)
        has_assign = 'assign' in token_set
        
        # Check for ternary operator (behavioral)
        has_ternary = '?' in code_clean
        
        if behavioral_count > 0:
            return 'behavioral'
        elif gate_level_count > 0:
            return 'gate-level'
        elif has_assign:
            if has_ternary:
                return 'behavioral-continuous'
            else:
                # Check if using &, |, ~ operators (gate-level style)
                if re.search(r'[&|~^]', code_clean):
                    return 'gate-level-continuous'
                else:
                    return 'continuous-assignment'
        else:
            return 'structural'
    
    def _validate_interface(
        self,
        submitted_module: Dict,
        expected_name: str,
        expected_inputs: List[str],
        expected_outputs: List[str],
        compliance: ArchitecturalCompliance
    ):
        """Validate module interface matches specification"""
        # Check module name
        if submitted_module['name'] != expected_name:
            compliance.add_violation(
                f"Module name mismatch: expected '{expected_name}', got '{submitted_module['name']}'"
            )
            compliance.interface_compliance = False
        
        # Check inputs
        submitted_inputs = set(submitted_module['inputs'])
        required_inputs = set(expected_inputs)
        
        missing_inputs = required_inputs - submitted_inputs
        extra_inputs = submitted_inputs - required_inputs
        
        if missing_inputs:
            compliance.add_violation(
                f"Missing required inputs: {', '.join(missing_inputs)}"
            )
            compliance.interface_compliance = False
        
        if extra_inputs:
            compliance.add_warning(
                f"Extra inputs not in specification: {', '.join(extra_inputs)}"
            )
        
        # Check outputs
        submitted_outputs = set(submitted_module['outputs'])
        required_outputs = set(expected_outputs)
        
        missing_outputs = required_outputs - submitted_outputs
        extra_outputs = submitted_outputs - required_outputs
        
        if missing_outputs:
            compliance.add_violation(
                f"Missing required outputs: {', '.join(missing_outputs)}"
            )
            compliance.interface_compliance = False
        
        if extra_outputs:
            compliance.add_warning(
                f"Extra outputs not in specification: {', '.join(extra_outputs)}"
            )
    
    def _validate_structural_style(
        self,
        verilog_code: str,
        required_style: str,
        detected_style: str,
        compliance: ArchitecturalCompliance
    ):
        """Validate implementation style matches requirements"""
        if required_style == 'gate-level':
            if 'behavioral' in detected_style:
                compliance.add_violation(
                    f"Design spec requires gate-level implementation, but detected {detected_style} constructs"
                )
                compliance.structural_compliance = False
            elif detected_style == 'continuous-assignment':
                compliance.add_warning(
                    "Using continuous assignment without explicit gate-level operators - verify compliance"
                )
        
        elif required_style == 'behavioral':
            if detected_style in ['gate-level', 'gate-level-continuous']:
                compliance.add_warning(
                    f"Design allows behavioral constructs, but using {detected_style} (this may be fine)"
                )
        
        elif required_style == 'structural':
            if detected_style == 'behavioral':
                compliance.add_violation(
                    "Design spec requires structural implementation, but using behavioral constructs"
                )
                compliance.structural_compliance = False
    
    def _validate_design_constraints(
        self,
        verilog_code: str,
        constraints: Dict,
        compliance: ArchitecturalCompliance
    ):
        """Validate specific design constraints are met"""
        code_lower = verilog_code.lower()
        
        # Check for common constraint violations
        explicit_constraints = constraints.get('explicit_constraints', [])
        
        for constraint in explicit_constraints:
            # Check for combinational requirement
            if 'combinational' in constraint and 'sequential' not in constraint:
                if re.search(r'\balways\s*@\s*\(\s*posedge', verilog_code):
                    compliance.add_violation(
                        "Design must be combinational (no clocking), but found sequential logic"
                    )
                    compliance.constraint_compliance = False
            
            # Check for no memory elements
            if 'no memory' in constraint or 'no register' in constraint:
                if 'reg' in code_lower or 'always' in code_lower:
                    compliance.add_warning(
                        "Design constraint specifies no memory elements - verify implementation"
                    )
    
    def _check_forbidden_constructs(
        self,
        verilog_code: str,
        forbidden: List[str],
        compliance: ArchitecturalCompliance
    ):
        """Check for forbidden constructs in code"""
        code_lower = verilog_code.lower()
        
        for forbidden_item in forbidden:
            if forbidden_item == 'behavioral' and 'always' in code_lower:
                compliance.add_violation(
                    "Behavioral constructs (always blocks) are forbidden by design spec"
                )
                compliance.constraint_compliance = False
            
            elif forbidden_item == 'latch':
                # Simple latch detection heuristic
                if 'always' in code_lower and '@' in code_lower:
                    if not re.search(r'posedge|negedge', code_lower):
                        compliance.add_warning(
                            "Potential latch detected - verify all signals are assigned in all paths"
                        )
    
    def _generate_validation_log(
        self,
        compliance: ArchitecturalCompliance,
        constraints: Dict,
        specs: Dict
    ) -> str:
        """Generate human-readable validation log"""
        log_lines = ["Architectural Validation Report", "=" * 40]
        
        log_lines.append(f"\nRequired Style: {compliance.required_style or 'Not specified'}")
        log_lines.append(f"Detected Style: {compliance.detected_style or 'Unknown'}")
        log_lines.append(f"\nCompliance Score: {compliance.compliance_score:.2f}")
        log_lines.append(f"Overall Compliant: {'YES' if compliance.is_compliant else 'NO'}")
        
        if compliance.violations:
            log_lines.append(f"\n❌ Violations ({len(compliance.violations)}):")
            for i, violation in enumerate(compliance.violations, 1):
                log_lines.append(f"  {i}. {violation}")
        
        if compliance.warnings:
            log_lines.append(f"\n⚠️  Warnings ({len(compliance.warnings)}):")
            for i, warning in enumerate(compliance.warnings, 1):
                log_lines.append(f"  {i}. {warning}")
        
        if compliance.is_compliant and not compliance.warnings:
            log_lines.append("\n✓ No violations or warnings detected")
        
        return '\n'.join(log_lines)


# Singleton validator instance
_validator: Optional[ArchitecturalValidator] = None


def get_validator() -> ArchitecturalValidator:
    """Get or create the singleton validator instance"""
    global _validator
    if _validator is None:
        _validator = ArchitecturalValidator()
    return _validator
