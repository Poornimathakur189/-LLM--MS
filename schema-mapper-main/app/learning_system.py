"""
Learning System - Remembers and reuses accepted fixes for future files.
"""

import json
import os
import pickle
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class LearningSystem:
    """Learning system that remembers and reuses accepted fixes."""
    
    def __init__(self, learning_data_path: str = "learning_data.json"):
        self.learning_data_path = Path(learning_data_path)
        self.learning_data = {
            'mapping_patterns': {},  # Learned column mapping patterns
            'fix_patterns': {},      # Learned fix patterns
            'user_preferences': {},  # User preferences and overrides
            'success_metrics': {},   # Success metrics for different approaches
            'last_updated': None
        }
        
        # Learning parameters
        self.min_confidence_threshold = 0.7
        self.min_occurrences_for_learning = 3
        self.max_learning_history = 1000
        
        # Load existing learning data
        self.load_learning_data()
    
    def load_learning_data(self) -> bool:
        """Load learning data from file."""
        try:
            if self.learning_data_path.exists():
                with open(self.learning_data_path, 'r') as f:
                    self.learning_data = json.load(f)
                logger.info(f"Loaded learning data from {self.learning_data_path}")
                return True
            else:
                logger.info("No existing learning data found, starting fresh")
                return False
        except Exception as e:
            logger.error(f"Error loading learning data: {e}")
            return False
    
    def save_learning_data(self) -> bool:
        """Save learning data to file."""
        try:
            self.learning_data['last_updated'] = datetime.now().isoformat()
            
            # Create directory if it doesn't exist
            self.learning_data_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.learning_data_path, 'w') as f:
                json.dump(self.learning_data, f, indent=2)
            
            logger.info(f"Saved learning data to {self.learning_data_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving learning data: {e}")
            return False
    
    def learn_from_mapping(self, source_header: str, canonical_header: str, confidence: float, user_override: bool = False) -> None:
        """
        Learn from a column mapping decision.
        
        Args:
            source_header: Source column header
            canonical_header: Canonical column header
            confidence: Confidence score of the mapping
            user_override: Whether this was a user override
        """
        try:
            mapping_key = source_header.lower().strip()
            
            if mapping_key not in self.learning_data['mapping_patterns']:
                self.learning_data['mapping_patterns'][mapping_key] = {
                    'canonical_mappings': Counter(),
                    'confidence_scores': [],
                    'user_overrides': 0,
                    'total_occurrences': 0,
                    'last_used': None
                }
            
            pattern = self.learning_data['mapping_patterns'][mapping_key]
            
            # Update mapping frequency
            pattern['canonical_mappings'][canonical_header] += 1
            pattern['confidence_scores'].append(confidence)
            pattern['total_occurrences'] += 1
            pattern['last_used'] = datetime.now().isoformat()
            
            if user_override:
                pattern['user_overrides'] += 1
            
            # Keep only recent confidence scores (last 100)
            if len(pattern['confidence_scores']) > 100:
                pattern['confidence_scores'] = pattern['confidence_scores'][-100:]
            
            logger.info(f"Learned mapping: {source_header} -> {canonical_header} (confidence: {confidence})")
            
        except Exception as e:
            logger.error(f"Error learning from mapping: {e}")
    
    def learn_from_fix(self, fix_id: str, accepted: bool, custom_value: Optional[str] = None) -> None:
        """
        Learn from a fix application decision.
        
        Args:
            fix_id: Unique identifier for the fix
            accepted: Whether the fix was accepted
            custom_value: Custom value provided by the user
        """
        try:
            if fix_id not in self.learning_data['fix_patterns']:
                self.learning_data['fix_patterns'][fix_id] = {
                    'acceptance_rate': 0.0,
                    'total_applications': 0,
                    'accepted_count': 0,
                    'custom_values': [],
                    'last_applied': None,
                    'effectiveness_score': 0.0
                }
            
            fix_pattern = self.learning_data['fix_patterns'][fix_id]
            
            # Update statistics
            fix_pattern['total_applications'] += 1
            fix_pattern['last_applied'] = datetime.now().isoformat()
            
            if accepted:
                fix_pattern['accepted_count'] += 1
                if custom_value:
                    fix_pattern['custom_values'].append(custom_value)
            
            # Calculate acceptance rate
            fix_pattern['acceptance_rate'] = fix_pattern['accepted_count'] / fix_pattern['total_applications']
            
            # Keep only recent custom values (last 50)
            if len(fix_pattern['custom_values']) > 50:
                fix_pattern['custom_values'] = fix_pattern['custom_values'][-50:]
            
            logger.info(f"Learned from fix {fix_id}: accepted={accepted}, acceptance_rate={fix_pattern['acceptance_rate']:.2f}")
            
        except Exception as e:
            logger.error(f"Error learning from fix: {e}")
    
    def get_learned_mapping(self, source_header: str) -> Optional[Tuple[str, float]]:
        """
        Get learned mapping for a source header.
        
        Args:
            source_header: Source column header
            
        Returns:
            Tuple of (canonical_header, confidence) or None if no learning
        """
        try:
            mapping_key = source_header.lower().strip()
            
            if mapping_key not in self.learning_data['mapping_patterns']:
                return None
            
            pattern = self.learning_data['mapping_patterns'][mapping_key]
            
            # Check if we have enough occurrences to trust this learning
            if pattern['total_occurrences'] < self.min_occurrences_for_learning:
                return None
            
            # Get the most common mapping
            if not pattern['canonical_mappings']:
                return None
            
            most_common = pattern['canonical_mappings'].most_common(1)[0]
            canonical_header = most_common[0]
            frequency = most_common[1]
            
            # Calculate confidence based on frequency and historical confidence
            frequency_confidence = min(frequency / pattern['total_occurrences'], 1.0)
            
            if pattern['confidence_scores']:
                avg_confidence = sum(pattern['confidence_scores']) / len(pattern['confidence_scores'])
                combined_confidence = (frequency_confidence + avg_confidence) / 2
            else:
                combined_confidence = frequency_confidence
            
            # Boost confidence if user has overridden this mapping
            if pattern['user_overrides'] > 0:
                combined_confidence = min(combined_confidence + 0.1, 1.0)
            
            if combined_confidence >= self.min_confidence_threshold:
                return canonical_header, combined_confidence
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting learned mapping: {e}")
            return None
    
    def get_learned_fixes(self, issue_type: str) -> List[Dict[str, Any]]:
        """
        Get learned fixes for a specific issue type.
        
        Args:
            issue_type: Type of issue
            
        Returns:
            List of learned fix patterns
        """
        try:
            learned_fixes = []
            
            for fix_id, pattern in self.learning_data['fix_patterns'].items():
                # Filter by issue type (this would need to be stored with the fix)
                if pattern.get('issue_type') == issue_type:
                    if pattern['total_applications'] >= self.min_occurrences_for_learning:
                        learned_fixes.append({
                            'fix_id': fix_id,
                            'acceptance_rate': pattern['acceptance_rate'],
                            'total_applications': pattern['total_applications'],
                            'effectiveness_score': pattern['effectiveness_score'],
                            'common_custom_values': self._get_common_custom_values(pattern['custom_values'])
                        })
            
            # Sort by effectiveness and acceptance rate
            learned_fixes.sort(key=lambda x: (x['effectiveness_score'], x['acceptance_rate']), reverse=True)
            
            return learned_fixes
            
        except Exception as e:
            logger.error(f"Error getting learned fixes: {e}")
            return []
    
    def _get_common_custom_values(self, custom_values: List[str]) -> List[Tuple[str, int]]:
        """Get most common custom values."""
        if not custom_values:
            return []
        
        value_counts = Counter(custom_values)
        return value_counts.most_common(5)
    
    def update_success_metrics(self, operation_type: str, success: bool, metrics: Dict[str, Any]) -> None:
        """
        Update success metrics for different operations.
        
        Args:
            operation_type: Type of operation (e.g., 'mapping', 'fixing', 'validation')
            success: Whether the operation was successful
            metrics: Additional metrics
        """
        try:
            if operation_type not in self.learning_data['success_metrics']:
                self.learning_data['success_metrics'][operation_type] = {
                    'total_operations': 0,
                    'successful_operations': 0,
                    'success_rate': 0.0,
                    'average_processing_time': 0.0,
                    'last_updated': None
                }
            
            metric = self.learning_data['success_metrics'][operation_type]
            metric['total_operations'] += 1
            
            if success:
                metric['successful_operations'] += 1
            
            metric['success_rate'] = metric['successful_operations'] / metric['total_operations']
            metric['last_updated'] = datetime.now().isoformat()
            
            # Update average processing time if provided
            if 'processing_time' in metrics:
                current_avg = metric['average_processing_time']
                total_ops = metric['total_operations']
                new_time = metrics['processing_time']
                
                metric['average_processing_time'] = ((current_avg * (total_ops - 1)) + new_time) / total_ops
            
            logger.info(f"Updated success metrics for {operation_type}: success_rate={metric['success_rate']:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating success metrics: {e}")
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get a summary of the learning system's knowledge."""
        try:
            summary = {
                'total_mapping_patterns': len(self.learning_data['mapping_patterns']),
                'total_fix_patterns': len(self.learning_data['fix_patterns']),
                'last_updated': self.learning_data.get('last_updated'),
                'mapping_patterns_summary': {},
                'fix_patterns_summary': {},
                'success_metrics': self.learning_data['success_metrics']
            }
            
            # Summarize mapping patterns
            for pattern_key, pattern in self.learning_data['mapping_patterns'].items():
                if pattern['total_occurrences'] >= self.min_occurrences_for_learning:
                    most_common = pattern['canonical_mappings'].most_common(1)[0]
                    summary['mapping_patterns_summary'][pattern_key] = {
                        'canonical_mapping': most_common[0],
                        'frequency': most_common[1],
                        'total_occurrences': pattern['total_occurrences'],
                        'user_overrides': pattern['user_overrides']
                    }
            
            # Summarize fix patterns
            for fix_id, pattern in self.learning_data['fix_patterns'].items():
                if pattern['total_applications'] >= self.min_occurrences_for_learning:
                    summary['fix_patterns_summary'][fix_id] = {
                        'acceptance_rate': pattern['acceptance_rate'],
                        'total_applications': pattern['total_applications'],
                        'effectiveness_score': pattern['effectiveness_score']
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting learning summary: {e}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 90) -> None:
        """Clean up old learning data to prevent the system from growing too large."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            cutoff_iso = cutoff_date.isoformat()
            
            # Clean up old mapping patterns
            patterns_to_remove = []
            for pattern_key, pattern in self.learning_data['mapping_patterns'].items():
                if pattern.get('last_used', '') < cutoff_iso:
                    patterns_to_remove.append(pattern_key)
            
            for pattern_key in patterns_to_remove:
                del self.learning_data['mapping_patterns'][pattern_key]
            
            # Clean up old fix patterns
            fixes_to_remove = []
            for fix_id, pattern in self.learning_data['fix_patterns'].items():
                if pattern.get('last_applied', '') < cutoff_iso:
                    fixes_to_remove.append(fix_id)
            
            for fix_id in fixes_to_remove:
                del self.learning_data['fix_patterns'][fix_id]
            
            if patterns_to_remove or fixes_to_remove:
                logger.info(f"Cleaned up {len(patterns_to_remove)} mapping patterns and {len(fixes_to_remove)} fix patterns")
                self.save_learning_data()
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
    
    def export_learning_data(self, export_path: str) -> bool:
        """Export learning data to a file."""
        try:
            export_path = Path(export_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(export_path, 'w') as f:
                json.dump(self.learning_data, f, indent=2)
            
            logger.info(f"Exported learning data to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting learning data: {e}")
            return False
    
    def import_learning_data(self, import_path: str) -> bool:
        """Import learning data from a file."""
        try:
            import_path = Path(import_path)
            
            if not import_path.exists():
                logger.error(f"Import file not found: {import_path}")
                return False
            
            with open(import_path, 'r') as f:
                imported_data = json.load(f)
            
            # Merge with existing data
            self.learning_data.update(imported_data)
            self.save_learning_data()
            
            logger.info(f"Imported learning data from {import_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing learning data: {e}")
            return False