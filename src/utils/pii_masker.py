"""
PII Masker Utility
Masks sensitive information before AI processing
"""

import re
import hashlib
import logging
from typing import Dict, Tuple, List
from datetime import datetime

logger = logging.getLogger(__name__)

class PIIMasker:
    """
    Masks personally identifiable information in legal documents
    Ensures compliance before sending to external AI services
    """
    
    def __init__(self, encryption_key: str = "default-secure-key"):
        """Initialize PII masker"""
        
        self.encryption_key = encryption_key
        self.masking_map = {}
        self.pattern_cache = {}
        
        # Compile patterns once for performance
        self._compile_patterns()
        
        logger.info("PII masker initialized")
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency"""
        
        self.patterns = {
            'ssn': re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
            'case_number': re.compile(
                r'\b(\d{2,4}-[A-Z]{2,4}-\d{4,6}|Case\s+No\.?\s*[:.]?\s*[\w\-]+)\b',
                re.IGNORECASE
            ),
            'phone': re.compile(
                # Match various phone formats but avoid plain 10-digit numbers
                r'\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b'
            ),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'credit_card': re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
            'address': re.compile(
                r'\b\d+\s+[\w\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b',
                re.IGNORECASE
            ),
            'account_number': re.compile(
                # Match account numbers with context
                r'\b(?:Account\s*(?:Number|#|No\.?)[\s:]*)?(\d{8,12})\b',
                re.IGNORECASE
            )
        }
    
    def mask_document(self, text: str) -> Tuple[str, Dict]:
        """
        Mask all PII in document
        
        Args:
            text: Original document text
            
        Returns:
            Tuple of (masked_text, masking_map)
        """
        
        self.masking_map = {}
        masked_text = text
        
        # Apply masking in order of priority
        masked_text = self._mask_pattern(masked_text, 'ssn', self._mask_ssn)
        masked_text = self._mask_pattern(masked_text, 'credit_card', self._mask_credit_card)
        masked_text = self._mask_pattern(masked_text, 'case_number', self._mask_case_number)
        masked_text = self._mask_pattern(masked_text, 'account_number', self._mask_account_number)
        masked_text = self._mask_pattern(masked_text, 'phone', self._mask_phone)
        masked_text = self._mask_pattern(masked_text, 'email', self._mask_email)
        masked_text = self._mask_pattern(masked_text, 'address', self._mask_address)
        masked_text = self._mask_party_names(masked_text)
        
        return masked_text, self.masking_map
    
    def _mask_pattern(self, text: str, pattern_name: str, mask_func) -> str:
        """Generic pattern masking"""
        
        pattern = self.patterns.get(pattern_name)
        if not pattern:
            return text
        
        return pattern.sub(mask_func, text)
    
    def _mask_ssn(self, match) -> str:
        """Mask SSN keeping last 4 digits"""
        
        ssn = match.group()
        digits = re.sub(r'[^\d]', '', ssn)
        
        if len(digits) == 9:
            last_four = digits[-4:]
            token = f"[SSN-XXX-XX-{last_four}]"
            
            self.masking_map[token] = {
                "original": ssn,
                "type": "SSN",
                "hash": self._hash(ssn)
            }
            
            return token
        
        return ssn
    
    def _mask_credit_card(self, match) -> str:
        """Mask credit card with Luhn validation"""
        
        card = match.group()
        digits = re.sub(r'[^\d]', '', card)
        
        if self._is_valid_credit_card(digits):
            last_four = digits[-4:]
            token = f"[CARD-****-{last_four}]"
            
            self.masking_map[token] = {
                "original": card,
                "type": "CREDIT_CARD",
                "hash": self._hash(card)
            }
            
            return token
        
        return card
    
    def _is_valid_credit_card(self, digits: str) -> bool:
        """Validate credit card using Luhn algorithm"""
        
        if len(digits) < 13 or len(digits) > 19:
            return False
        
        total = 0
        for i, digit in enumerate(reversed(digits)):
            n = int(digit)
            if i % 2 == 1:
                n *= 2
                if n > 9:
                    n -= 9
            total += n
        
        return total % 10 == 0
    
    def _mask_case_number(self, match) -> str:
        """Mask case numbers"""
        
        case = match.group()
        # Generate sequential ID
        case_id = len([k for k in self.masking_map if k.startswith('[CASE-')]) + 1
        token = f"[CASE-{case_id:03d}]"
        
        self.masking_map[token] = {
            "original": case,
            "type": "CASE_NUMBER",
            "hash": self._hash(case)
        }
        
        return token
    
    def _mask_account_number(self, match) -> str:
        """Mask account numbers"""
        
        account = match.group()
        
        acct_id = len([k for k in self.masking_map if k.startswith('[ACCOUNT-')]) + 1
        token = f"[ACCOUNT-{acct_id:03d}]"
        
        self.masking_map[token] = {
            "original": account,
            "type": "ACCOUNT_NUMBER",
            "hash": self._hash(account)
        }
        
        return token
    
    def _mask_phone(self, match) -> str:
        """Mask phone numbers"""
        
        phone = match.group()
        phone_id = len([k for k in self.masking_map if k.startswith('[PHONE-')]) + 1
        token = f"[PHONE-{phone_id:03d}]"
        
        self.masking_map[token] = {
            "original": phone,
            "type": "PHONE",
            "hash": self._hash(phone)
        }
        
        return token
    
    def _mask_email(self, match) -> str:
        """Mask email addresses"""
        
        email = match.group()
        domain = email.split('@')[1] if '@' in email else "unknown"
        email_id = len([k for k in self.masking_map if k.startswith('[EMAIL-')]) + 1
        # Don't include @ in token to avoid matching email regex
        token = f"[EMAIL-{email_id:03d}-{domain.replace('.', '_')}]"
        
        self.masking_map[token] = {
            "original": email,
            "type": "EMAIL",
            "hash": self._hash(email)
        }
        
        return token
    
    def _mask_address(self, match) -> str:
        """Mask street addresses"""
        
        address = match.group()
        addr_id = len([k for k in self.masking_map if k.startswith('[ADDRESS-')]) + 1
        token = f"[ADDRESS-{addr_id:03d}]"
        
        self.masking_map[token] = {
            "original": address,
            "type": "ADDRESS",
            "hash": self._hash(address)
        }
        
        return token
    
    def _mask_party_names(self, text: str) -> str:
        """Mask party names in legal documents"""
        
        # Legal party indicators
        party_patterns = [
            (r'Plaintiff[:]?\s*([A-Z][a-zA-Z\s]+?)(?:\n|,|v\.)', 'PLAINTIFF'),
            (r'Defendant[:]?\s*([A-Z][a-zA-Z\s]+?)(?:\n|,|v\.)', 'DEFENDANT'),
            (r'Petitioner[:]?\s*([A-Z][a-zA-Z\s]+?)(?:\n|,)', 'PETITIONER'),
            (r'Respondent[:]?\s*([A-Z][a-zA-Z\s]+?)(?:\n|,)', 'RESPONDENT')
        ]
        
        masked = text
        party_counters = {}
        
        for pattern, party_type in party_patterns:
            matches = re.finditer(pattern, masked, re.IGNORECASE)
            
            for match in matches:
                name = match.group(1).strip()
                
                if party_type not in party_counters:
                    party_counters[party_type] = 0
                
                party_counters[party_type] += 1
                token = f"[{party_type}-{party_counters[party_type]}]"
                
                self.masking_map[token] = {
                    "original": name,
                    "type": party_type,
                    "hash": self._hash(name)
                }
                
                masked = masked.replace(name, token)
        
        return masked
    
    def unmask_document(self, masked_text: str, masking_map: Dict) -> str:
        """
        Restore original values in masked text
        
        Args:
            masked_text: Text with masked tokens
            masking_map: Mapping of tokens to original values
            
        Returns:
            Original text with values restored
        """
        
        unmasked = masked_text
        
        for token, info in masking_map.items():
            original = info.get('original', token)
            unmasked = unmasked.replace(token, original)
        
        return unmasked
    
    def _hash(self, value: str) -> str:
        """Generate secure hash for verification"""
        
        salted = f"{self.encryption_key}:{value}"
        return hashlib.sha256(salted.encode()).hexdigest()[:16]
    
    def validate_masking(self, original: str, masked: str) -> Dict:
        """
        Validate masking effectiveness
        
        Returns:
            Validation results with any potential leaks
        """
        
        results = {
            "success": True,
            "potential_leaks": [],
            "entities_masked": len(self.masking_map),
            "masking_percentage": 0.0
        }
        
        # Check for remaining patterns
        for pattern_name, pattern in self.patterns.items():
            matches = pattern.findall(masked)
            if matches:
                results["success"] = False
                results["potential_leaks"].append({
                    "type": pattern_name,
                    "count": len(matches),
                    "samples": matches[:3]
                })
        
        # Calculate masking percentage
        if original:
            results["masking_percentage"] = (
                1.0 - len(masked) / len(original)
            ) * 100
        
        return results