"""
Logger Utility
Centralized logging configuration
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
import json

def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Setup logger with consistent formatting
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger
    """
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler
    logger.addHandler(console_handler)
    
    # File handler for errors
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    error_handler = logging.FileHandler(
        log_dir / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)
    
    return logger

class AuditLogger:
    """
    Audit logger for compliance tracking
    """
    
    def __init__(self, log_file: str = "audit.jsonl"):
        """Initialize audit logger"""
        
        self.log_file = Path("logs") / log_file
        self.log_file.parent.mkdir(exist_ok=True)
    
    def log(self, event: str, data: dict):
        """
        Log audit event
        
        Args:
            event: Event type
            data: Event data
        """
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "data": data
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')