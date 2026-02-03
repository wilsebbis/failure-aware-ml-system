"""
Memory-Optimized Data Loading Utilities.

Provides strategies for handling large datasets (1GB+) without OOM:
1. Chunked loading - Process data in batches
2. Column selection - Load only essential columns
3. Dtype optimization - Use memory-efficient types
4. Dask integration - Optional parallel out-of-core processing

Usage:
    from src.data.memory_loader import MemoryOptimizedLoader
    
    loader = MemoryOptimizedLoader(
        filepath="data.csv",
        essential_cols=["col1", "col2"],
        mode="chunked"  # or "dask" or "optimized"
    )
    df = loader.load()
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
import logging

logger = logging.getLogger(__name__)


# Optimal dtypes for common column patterns
DTYPE_OPTIMIZATIONS: Dict[str, Any] = {
    # Integer columns (use smallest that fits)
    "int_small": "int16",      # -32k to 32k
    "int_medium": "int32",     # -2B to 2B
    # Float columns
    "float": "float32",        # 6-7 decimal precision
    # Categorical (high cardinality reduction)
    "category": "category",
}


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame dtypes to reduce memory usage.
    
    Strategies:
    - Convert float64 -> float32 (50% reduction)
    - Convert int64 -> int32/int16 (50-75% reduction)
    - Convert object -> category (90%+ reduction for low cardinality)
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with optimized dtypes
    """
    initial_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type == "float64":
            # Downcast floats
            df[col] = df[col].astype("float32")
            
        elif col_type == "int64":
            # Downcast integers based on range
            col_min, col_max = df[col].min(), df[col].max()
            if col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype("int16")
            elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype("int32")
                
        elif col_type == "object":
            # Convert to category if low cardinality
            n_unique = df[col].nunique()
            n_total = len(df)
            if n_unique / n_total < 0.5:  # Less than 50% unique
                df[col] = df[col].astype("category")
    
    final_memory = df.memory_usage(deep=True).sum() / 1024**2
    reduction = (1 - final_memory / initial_memory) * 100
    logger.info(f"Memory optimization: {initial_memory:.1f}MB -> {final_memory:.1f}MB ({reduction:.1f}% reduction)")
    
    return df


class MemoryOptimizedLoader:
    """
    Memory-optimized CSV loader supporting multiple strategies.
    
    Modes:
        - "standard": Basic pandas load with column selection + dtype optimization
        - "chunked": Process in chunks, useful for transformation pipelines
        - "dask": Parallel out-of-core (requires dask[dataframe])
        
    Example:
        loader = MemoryOptimizedLoader(
            filepath="large_file.csv",
            essential_cols=["col1", "col2", "target"],
            mode="chunked",
            chunk_size=50000
        )
        df = loader.load()
    """
    
    def __init__(
        self,
        filepath: Path,
        essential_cols: Optional[List[str]] = None,
        mode: str = "standard",  # "standard", "chunked", "dask"
        chunk_size: int = 50000,
        nrows: Optional[int] = None,
        dtype_hints: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize loader.
        
        Args:
            filepath: Path to CSV file
            essential_cols: Columns to load (None = all)
            mode: Loading mode ("standard", "chunked", "dask")
            chunk_size: Rows per chunk for chunked mode
            nrows: Max rows to load (None = all)
            dtype_hints: Optional dtype specifications
        """
        self.filepath = Path(filepath)
        self.essential_cols = essential_cols
        self.mode = mode
        self.chunk_size = chunk_size
        self.nrows = nrows
        self.dtype_hints = dtype_hints or {}
        
    def load(self) -> pd.DataFrame:
        """Load data using configured strategy."""
        if self.mode == "dask":
            return self._load_dask()
        elif self.mode == "chunked":
            return self._load_chunked()
        else:
            return self._load_standard()
    
    def _load_standard(self) -> pd.DataFrame:
        """Standard load with column selection and dtype optimization."""
        logger.info(f"Loading {self.filepath} (standard mode)")
        
        usecols = (lambda c: c in self.essential_cols) if self.essential_cols else None
        
        df = pd.read_csv(
            self.filepath,
            low_memory=False,
            nrows=self.nrows,
            usecols=usecols,
            dtype=self.dtype_hints if self.dtype_hints else None,
        )
        
        # Apply dtype optimization
        df = optimize_dtypes(df)
        
        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
        return df
    
    def _load_chunked(self) -> pd.DataFrame:
        """
        Chunked loading - processes data in batches.
        
        Useful when you need to apply transformations during load
        or when data doesn't fit in memory even with optimization.
        """
        logger.info(f"Loading {self.filepath} (chunked mode, {self.chunk_size:,} rows/chunk)")
        
        usecols = (lambda c: c in self.essential_cols) if self.essential_cols else None
        
        chunks = []
        rows_loaded = 0
        
        reader = pd.read_csv(
            self.filepath,
            low_memory=False,
            chunksize=self.chunk_size,
            usecols=usecols,
            dtype=self.dtype_hints if self.dtype_hints else None,
        )
        
        for i, chunk in enumerate(reader):
            # Optimize each chunk immediately to save memory
            chunk = optimize_dtypes(chunk)
            chunks.append(chunk)
            rows_loaded += len(chunk)
            
            if self.nrows and rows_loaded >= self.nrows:
                break
                
            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {rows_loaded:,} rows")
        
        # Concatenate optimized chunks
        df = pd.concat(chunks, ignore_index=True)
        
        if self.nrows:
            df = df.head(self.nrows)
        
        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns (from {len(chunks)} chunks)")
        return df
    
    def _load_dask(self) -> pd.DataFrame:
        """
        Dask-based parallel loading for out-of-core processing.
        
        Advantages:
        - Parallel I/O across CPU cores
        - Out-of-core: can process data larger than RAM
        - Lazy evaluation until .compute()
        
        Requires: pip install dask[dataframe] pyarrow
        """
        try:
            import dask.dataframe as dd
        except ImportError:
            logger.warning("Dask not installed, falling back to chunked mode")
            logger.warning("Install with: pip install dask[dataframe] pyarrow")
            return self._load_chunked()
        
        logger.info(f"Loading {self.filepath} (dask mode)")
        
        # Dask handles column selection differently
        ddf = dd.read_csv(
            self.filepath,
            blocksize="64MB",  # Process in 64MB blocks
            dtype=self.dtype_hints if self.dtype_hints else None,
        )
        
        # Select columns
        if self.essential_cols:
            available_cols = [c for c in self.essential_cols if c in ddf.columns]
            ddf = ddf[available_cols]
        
        # Limit rows if specified
        if self.nrows:
            ddf = ddf.head(self.nrows, compute=False)
        
        # Compute (materialize to pandas)
        df = ddf.compute()
        
        # Apply dtype optimization
        df = optimize_dtypes(df)
        
        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns (via Dask)")
        return df


def get_memory_stats(df: pd.DataFrame) -> Dict[str, float]:
    """Get memory usage statistics for a DataFrame."""
    memory_bytes = df.memory_usage(deep=True).sum()
    return {
        "bytes": memory_bytes,
        "kb": memory_bytes / 1024,
        "mb": memory_bytes / 1024**2,
        "gb": memory_bytes / 1024**3,
    }
