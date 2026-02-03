"""
Abstract Base Class for Risk Data Adapters.

This module defines the contract that all dataset adapters must implement,
ensuring consistent data format regardless of the underlying data source.

The Adapter Pattern enables:
- Separation of Concerns: Data cleaning logic ≠ model training logic
- Extensibility: Add new datasets without touching core training code
- Abstraction: "Risk Data" is abstract, adapters are concrete implementations
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class DataSplit:
    """Container for train/validation/test splits."""
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


@dataclass
class AdapterMetadata:
    """Metadata about the loaded dataset."""
    name: str
    num_samples: int
    num_features: int
    target_rate: float  # Positive class rate
    temporal_column: Optional[str] = None  # For time-based splits
    primary_key: Optional[str] = None


class RiskDataAdapter(ABC):
    """
    Abstract Interface for Risk Data.
    
    Enforces that every dataset adapter provides data in a standard format,
    enabling the training pipeline to be dataset-agnostic.
    
    Subclasses must implement:
        - load_raw(): Fetch data from source (disk, S3, API)
        - feature_engineer(): Apply dataset-specific transforms
        - get_features_and_target(): Return processed X and y
        - get_categorical_features(): Identify categorical columns
        - get_split_strategy(): Define how to split data
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize adapter with configuration.
        
        Args:
            config: Dictionary containing dataset-specific settings
                   (paths, thresholds, column mappings, etc.)
        """
        self.config = config
        self.df: Optional[pd.DataFrame] = None
        self._metadata: Optional[AdapterMetadata] = None
    
    @property
    def metadata(self) -> AdapterMetadata:
        """Return metadata about the loaded dataset."""
        if self._metadata is None:
            raise ValueError("Data not loaded. Call load_raw() first.")
        return self._metadata
    
    @abstractmethod
    def load_raw(self) -> None:
        """
        Load raw files from disk/S3/API.
        
        This method should:
        - Fetch all required data files
        - Perform initial joins if multiple tables
        - Store result in self.df
        - Populate self._metadata
        """
        pass
    
    @abstractmethod
    def feature_engineer(self) -> None:
        """
        Apply dataset-specific feature transformations.
        
        This method should:
        - Handle missing values specific to this dataset
        - Create derived features (aggregations, ratios, etc.)
        - Encode categorical variables if needed
        - NOT perform train/test split (handled separately)
        """
        pass
    
    @abstractmethod
    def get_features_and_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Return processed feature matrix and target vector.
        
        Returns:
            X: DataFrame of features (cleaned, engineered)
            y: Series of target values
        """
        pass
    
    @abstractmethod
    def get_categorical_features(self) -> List[str]:
        """
        Return list of column names to be treated as categorical.
        
        Returns:
            List of column names requiring categorical encoding
        """
        pass
    
    @abstractmethod
    def get_split_strategy(self) -> str:
        """
        Return the recommended split strategy for this dataset.
        
        Returns:
            One of: "random", "temporal", "stratified"
        """
        pass
    
    def get_temporal_column(self) -> Optional[str]:
        """
        Return the column to use for temporal splits (if applicable).
        
        Returns:
            Column name for time-based splitting, or None if not applicable
        """
        return None
    
    def validate(self) -> bool:
        """
        Validate that the adapter is properly configured.
        
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If configuration is invalid
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_raw() first.")
        
        X, y = self.get_features_and_target()
        
        if X.empty:
            raise ValueError("Feature matrix is empty")
        
        if len(X) != len(y):
            raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
        
        cat_features = self.get_categorical_features()
        missing_cats = set(cat_features) - set(X.columns)
        if missing_cats:
            raise ValueError(f"Categorical features not in X: {missing_cats}")
        
        return True
