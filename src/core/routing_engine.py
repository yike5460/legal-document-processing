"""
Routing Engine
Determines document and deadline routing based on type and confidence
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class RoutingEngine:
    """
    Intelligent routing engine that determines whether documents/deadlines
    should be auto-processed or sent for human review based on document type
    and confidence scores.
    """
    
    # Routing destinations
    ATTORNEY_REVIEW = "ATTORNEY_REVIEW"
    PARALEGAL_REVIEW = "PARALEGAL_REVIEW"
    AUTO_PROCESS = "AUTO_PROCESS"
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize routing engine with configuration"""
        
        self.config = config or {}
        
        # Confidence thresholds
        self.thresholds = {
            "auto_process": self.config.get("confidence_threshold_auto", 0.95),
            "paralegal_review": self.config.get("confidence_threshold_review", 0.85),
            "attorney_review": 0.0  # Everything below paralegal threshold
        }
        
        # Document type routing rules
        self.document_rules = {
            "court_order": {
                "always_route_to": self.ATTORNEY_REVIEW,
                "reason": "Court orders require attorney review regardless of confidence",
                "priority": "CRITICAL",
                "sla_hours": 2
            },
            "motion": {
                "confidence_based": True,
                "priority": "HIGH",
                "sla_hours": 4,
                "min_confidence_for_auto": 0.95
            },
            "notice": {
                "confidence_based": True,
                "priority": "STANDARD",
                "sla_hours": 8,
                "min_confidence_for_auto": 0.90
            },
            "complaint": {
                "confidence_based": True,
                "priority": "HIGH",
                "sla_hours": 4,
                "min_confidence_for_auto": 0.92
            },
            "answer": {
                "confidence_based": True,
                "priority": "HIGH",
                "sla_hours": 4,
                "min_confidence_for_auto": 0.92
            },
            "discovery": {
                "confidence_based": True,
                "priority": "HIGH",
                "sla_hours": 6,
                "min_confidence_for_auto": 0.90
            },
            "unknown": {
                "always_route_to": self.ATTORNEY_REVIEW,
                "reason": "Unknown document types require attorney review",
                "priority": "HIGH",
                "sla_hours": 2
            }
        }
        
        # Special conditions that override normal routing
        self.override_conditions = {
            "critical_deadline": {
                "route_to": self.ATTORNEY_REVIEW,
                "reason": "Critical deadlines require attorney review"
            },
            "invalid_validation": {
                "route_to": self.ATTORNEY_REVIEW,
                "reason": "Invalid deadline patterns require attorney review"
            },
            "calculation_error": {
                "route_to": self.ATTORNEY_REVIEW,
                "reason": "Calculation errors require attorney review"
            },
            "multiple_warnings": {
                "threshold": 2,
                "route_to": self.PARALEGAL_REVIEW,
                "reason": "Multiple warnings require human review"
            },
            "ambiguous_language": {
                "route_to": self.PARALEGAL_REVIEW,
                "reason": "Ambiguous language requires human interpretation"
            }
        }
        
        # Routing statistics
        self.stats = {
            "total_routed": 0,
            "attorney_review": 0,
            "paralegal_review": 0,
            "auto_process": 0,
            "override_applied": 0
        }
        
        logger.info("Routing engine initialized")
    
    def route_document(self, 
                      classification: Dict,
                      deadlines: List[Dict],
                      metadata: Optional[Dict] = None) -> Dict:
        """
        Determine routing for entire document.
        
        Args:
            classification: Document classification result
            deadlines: List of extracted deadlines with confidence
            metadata: Optional document metadata
            
        Returns:
            Document-level routing decision
        """
        
        doc_type = classification.get("primary_type", "unknown")
        doc_confidence = classification.get("confidence", 0.0)
        
        # Calculate overall deadline confidence
        if deadlines:
            deadline_confidences = [d.get("confidence", 0) for d in deadlines]
            avg_deadline_confidence = sum(deadline_confidences) / len(deadline_confidences)
            min_deadline_confidence = min(deadline_confidences)
        else:
            avg_deadline_confidence = 0.0
            min_deadline_confidence = 0.0
        
        # Combine document and deadline confidence
        overall_confidence = (doc_confidence * 0.3 + avg_deadline_confidence * 0.7)
        
        # Get document type rules
        doc_rules = self.document_rules.get(doc_type, self.document_rules["unknown"])
        
        # Check if document type has mandatory routing
        if "always_route_to" in doc_rules:
            routing = {
                "destination": doc_rules["always_route_to"],
                "reason": doc_rules["reason"],
                "priority": doc_rules["priority"],
                "sla_hours": doc_rules["sla_hours"],
                "confidence": overall_confidence,
                "override_applied": False
            }
        else:
            # Apply confidence-based routing
            min_auto_confidence = doc_rules.get("min_confidence_for_auto", 0.95)
            
            if overall_confidence >= min_auto_confidence and min_deadline_confidence >= 0.90:
                destination = self.AUTO_PROCESS
                reason = f"High confidence ({overall_confidence:.1%}) for {doc_type}"
            elif overall_confidence >= self.thresholds["paralegal_review"]:
                destination = self.PARALEGAL_REVIEW
                reason = f"Medium confidence ({overall_confidence:.1%}) requires verification"
            else:
                destination = self.ATTORNEY_REVIEW
                reason = f"Low confidence ({overall_confidence:.1%}) requires attorney review"
            
            routing = {
                "destination": destination,
                "reason": reason,
                "priority": doc_rules["priority"],
                "sla_hours": doc_rules["sla_hours"],
                "confidence": overall_confidence,
                "override_applied": False
            }
        
        # Check for override conditions
        override = self._check_overrides(deadlines, metadata)
        if override:
            routing["destination"] = override["route_to"]
            routing["reason"] = override["reason"]
            routing["override_applied"] = True
            routing["override_type"] = override["type"]
        
        # Update statistics
        self._update_stats(routing["destination"], routing["override_applied"])
        
        # Add routing metadata
        routing["metadata"] = {
            "document_type": doc_type,
            "document_confidence": doc_confidence,
            "deadline_count": len(deadlines),
            "avg_deadline_confidence": avg_deadline_confidence,
            "min_deadline_confidence": min_deadline_confidence,
            "timestamp": datetime.now().isoformat()
        }
        
        return routing
    
    def route_deadline(self,
                      deadline: Dict,
                      document_type: str,
                      document_routing: Optional[Dict] = None) -> Dict:
        """
        Determine routing for individual deadline.
        
        Args:
            deadline: Deadline with confidence and validation
            document_type: Type of document
            document_routing: Optional document-level routing decision
            
        Returns:
            Deadline-level routing decision
        """
        
        confidence = deadline.get("confidence", 0.0)
        validation_status = deadline.get("validation_status", "unknown")
        priority = deadline.get("priority", "standard")
        
        # Get document rules
        doc_rules = self.document_rules.get(document_type, self.document_rules["unknown"])
        
        # Start with document-level routing if available
        if document_routing:
            destination = document_routing["destination"]
            reason = f"Following document-level routing"
        else:
            # Apply deadline-specific routing
            if document_type == "court_order":
                destination = self.ATTORNEY_REVIEW
                reason = "Court order deadlines require attorney review"
            elif validation_status == "invalid":
                destination = self.ATTORNEY_REVIEW
                reason = "Invalid deadline pattern requires attorney review"
            elif priority == "critical" and confidence < 0.98:
                destination = self.ATTORNEY_REVIEW
                reason = "Critical deadline requires attorney review"
            elif confidence >= self.thresholds["auto_process"]:
                destination = self.AUTO_PROCESS
                reason = f"High confidence ({confidence:.1%})"
            elif confidence >= self.thresholds["paralegal_review"]:
                destination = self.PARALEGAL_REVIEW
                reason = f"Medium confidence ({confidence:.1%})"
            else:
                destination = self.ATTORNEY_REVIEW
                reason = f"Low confidence ({confidence:.1%})"
        
        # Check for deadline-specific overrides
        if deadline.get("calculation_error"):
            destination = self.ATTORNEY_REVIEW
            reason = "Calculation error requires attorney review"
        elif len(deadline.get("warnings", [])) > 2:
            if destination == self.AUTO_PROCESS:
                destination = self.PARALEGAL_REVIEW
                reason = "Multiple warnings require human review"
        
        return {
            "deadline_id": deadline.get("id", "unknown"),
            "destination": destination,
            "reason": reason,
            "confidence": confidence,
            "validation_status": validation_status,
            "priority": priority,
            "document_type": document_type
        }
    
    def _check_overrides(self, 
                        deadlines: List[Dict],
                        metadata: Optional[Dict] = None) -> Optional[Dict]:
        """Check if any override conditions apply"""
        
        # Check for critical deadlines
        critical_count = sum(
            1 for d in deadlines 
            if d.get("priority") == "critical"
        )
        if critical_count > 0:
            return {
                "type": "critical_deadline",
                "route_to": self.override_conditions["critical_deadline"]["route_to"],
                "reason": self.override_conditions["critical_deadline"]["reason"]
            }
        
        # Check for invalid validations
        invalid_count = sum(
            1 for d in deadlines 
            if d.get("validation_status") == "invalid"
        )
        if invalid_count > 0:
            return {
                "type": "invalid_validation",
                "route_to": self.override_conditions["invalid_validation"]["route_to"],
                "reason": self.override_conditions["invalid_validation"]["reason"]
            }
        
        # Check for calculation errors
        error_count = sum(
            1 for d in deadlines 
            if d.get("calculation_error")
        )
        if error_count > 0:
            return {
                "type": "calculation_error",
                "route_to": self.override_conditions["calculation_error"]["route_to"],
                "reason": self.override_conditions["calculation_error"]["reason"]
            }
        
        # Check for multiple warnings
        high_warning_count = sum(
            1 for d in deadlines 
            if len(d.get("warnings", [])) > self.override_conditions["multiple_warnings"]["threshold"]
        )
        if high_warning_count > 0:
            return {
                "type": "multiple_warnings",
                "route_to": self.override_conditions["multiple_warnings"]["route_to"],
                "reason": self.override_conditions["multiple_warnings"]["reason"]
            }
        
        # Check metadata for special conditions
        if metadata:
            if metadata.get("expedited"):
                return {
                    "type": "expedited",
                    "route_to": self.ATTORNEY_REVIEW,
                    "reason": "Expedited processing requires attorney review"
                }
            
            if metadata.get("vip_client"):
                return {
                    "type": "vip_client",
                    "route_to": self.ATTORNEY_REVIEW,
                    "reason": "VIP client documents require attorney review"
                }
        
        return None
    
    def _update_stats(self, destination: str, override_applied: bool):
        """Update routing statistics"""
        
        self.stats["total_routed"] += 1
        
        if destination == self.ATTORNEY_REVIEW:
            self.stats["attorney_review"] += 1
        elif destination == self.PARALEGAL_REVIEW:
            self.stats["paralegal_review"] += 1
        elif destination == self.AUTO_PROCESS:
            self.stats["auto_process"] += 1
        
        if override_applied:
            self.stats["override_applied"] += 1
    
    def get_statistics(self) -> Dict:
        """Get routing statistics"""
        
        total = self.stats["total_routed"]
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            "percentages": {
                "attorney_review": (self.stats["attorney_review"] / total) * 100,
                "paralegal_review": (self.stats["paralegal_review"] / total) * 100,
                "auto_process": (self.stats["auto_process"] / total) * 100,
                "override_rate": (self.stats["override_applied"] / total) * 100
            }
        }
    
    def explain_routing(self, routing_decision: Dict) -> str:
        """
        Generate human-readable explanation of routing decision.
        
        Args:
            routing_decision: Routing decision to explain
            
        Returns:
            Detailed explanation string
        """
        
        explanation = []
        
        destination = routing_decision["destination"]
        reason = routing_decision["reason"]
        confidence = routing_decision.get("confidence", 0)
        
        # Main routing explanation
        if destination == self.AUTO_PROCESS:
            explanation.append(
                f"This document will be AUTOMATICALLY PROCESSED because: {reason}"
            )
            explanation.append(
                f"The confidence level ({confidence:.1%}) exceeds the threshold for automation."
            )
        elif destination == self.PARALEGAL_REVIEW:
            explanation.append(
                f"This document requires PARALEGAL REVIEW because: {reason}"
            )
            explanation.append(
                f"The confidence level ({confidence:.1%}) indicates human verification is needed."
            )
        else:  # ATTORNEY_REVIEW
            explanation.append(
                f"This document requires ATTORNEY REVIEW because: {reason}"
            )
            explanation.append(
                "Legal expertise is required to properly handle this document."
            )
        
        # Add override information if applicable
        if routing_decision.get("override_applied"):
            override_type = routing_decision.get("override_type", "unknown")
            explanation.append(
                f"\nNOTE: Normal routing was overridden due to: {override_type}"
            )
        
        # Add priority and SLA information
        priority = routing_decision.get("priority", "STANDARD")
        sla = routing_decision.get("sla_hours", 24)
        explanation.append(
            f"\nPriority: {priority} | Response required within: {sla} hours"
        )
        
        # Add metadata if available
        metadata = routing_decision.get("metadata", {})
        if metadata:
            doc_type = metadata.get("document_type", "unknown")
            deadline_count = metadata.get("deadline_count", 0)
            explanation.append(
                f"Document type: {doc_type} | Deadlines found: {deadline_count}"
            )
        
        return "\n".join(explanation)