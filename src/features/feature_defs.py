"""
Feature Definitions

Domain-aligned feature definitions with documentation for auditability.
"""

from dataclasses import dataclass
from typing import Literal, Optional
from enum import Enum


class FeatureCategory(Enum):
    """Categories for feature organization."""
    DEMOGRAPHIC = "demographic"
    PAYMENT_HISTORY = "payment_history"
    BILL_AMOUNT = "bill_amount"
    PAYMENT_AMOUNT = "payment_amount"
    DERIVED = "derived"


@dataclass
class FeatureDefinition:
    """
    Definition of a single feature with audit metadata.
    
    Attributes:
        name: Feature column name
        category: Feature category for organization
        description: Human-readable description
        data_type: Expected data type
        domain_range: Valid range or values
        audit_justification: Why this feature is used (for audit trail)
        is_derived: Whether this is an engineered feature
        derivation_formula: Formula if derived
    """
    name: str
    category: FeatureCategory
    description: str
    data_type: Literal["continuous", "categorical", "ordinal", "binary"]
    domain_range: Optional[str] = None
    audit_justification: Optional[str] = None
    is_derived: bool = False
    derivation_formula: Optional[str] = None


# =============================================================================
# ORIGINAL FEATURES
# =============================================================================

DEMOGRAPHIC_FEATURES = [
    FeatureDefinition(
        name="LIMIT_BAL",
        category=FeatureCategory.DEMOGRAPHIC,
        description="Amount of given credit in NT dollars",
        data_type="continuous",
        domain_range="> 0",
        audit_justification="Credit limit is a strong indicator of creditworthiness assessment by the bank"
    ),
    FeatureDefinition(
        name="SEX",
        category=FeatureCategory.DEMOGRAPHIC,
        description="Gender (1=male, 2=female)",
        data_type="binary",
        domain_range="[1, 2]",
        audit_justification="Demographic feature for stratification analysis only"
    ),
    FeatureDefinition(
        name="AGE",
        category=FeatureCategory.DEMOGRAPHIC,
        description="Age in years",
        data_type="continuous",
        domain_range="[18, 100]",
        audit_justification="Age is correlated with financial stability and payment behavior"
    ),
]

PAYMENT_HISTORY_FEATURES = [
    FeatureDefinition(
        name=f"PAY_{i}",
        category=FeatureCategory.PAYMENT_HISTORY,
        description=f"Repayment status in month {6-idx if idx > 0 else 6} (-2=no consumption, -1=paid, 0=revolving, 1+=delay months)",
        data_type="ordinal",
        domain_range="[-2, 9]",
        audit_justification="Payment history is the strongest predictor of future default behavior"
    )
    for idx, i in enumerate([0, 2, 3, 4, 5, 6])
]

BILL_AMOUNT_FEATURES = [
    FeatureDefinition(
        name=f"BILL_AMT{i}",
        category=FeatureCategory.BILL_AMOUNT,
        description=f"Bill statement amount for month {7-i} (NT dollars)",
        data_type="continuous",
        domain_range="any (can be negative for overpayment)",
        audit_justification="Bill amounts indicate credit usage patterns and exposure"
    )
    for i in range(1, 7)
]

PAYMENT_AMOUNT_FEATURES = [
    FeatureDefinition(
        name=f"PAY_AMT{i}",
        category=FeatureCategory.PAYMENT_AMOUNT,
        description=f"Payment amount for month {7-i} (NT dollars)",
        data_type="continuous",
        domain_range=">= 0",
        audit_justification="Payment amounts show willingness and ability to repay"
    )
    for i in range(1, 7)
]


# =============================================================================
# DERIVED FEATURES
# =============================================================================

DERIVED_FEATURES = [
    FeatureDefinition(
        name="utilization_ratio",
        category=FeatureCategory.DERIVED,
        description="Credit utilization ratio (most recent bill / credit limit)",
        data_type="continuous",
        domain_range="[0, inf)",
        audit_justification="High utilization is a known risk factor for default",
        is_derived=True,
        derivation_formula="BILL_AMT1 / LIMIT_BAL"
    ),
    FeatureDefinition(
        name="avg_payment_delay",
        category=FeatureCategory.DERIVED,
        description="Average payment delay across 6 months",
        data_type="continuous",
        domain_range="[-2, 9]",
        audit_justification="Summarizes overall payment behavior pattern",
        is_derived=True,
        derivation_formula="mean(PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6)"
    ),
    FeatureDefinition(
        name="max_delay",
        category=FeatureCategory.DERIVED,
        description="Maximum payment delay recorded in 6 months",
        data_type="ordinal",
        domain_range="[-2, 9]",
        audit_justification="Peak delinquency is a strong default signal",
        is_derived=True,
        derivation_formula="max(PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6)"
    ),
    FeatureDefinition(
        name="has_severe_delay",
        category=FeatureCategory.DERIVED,
        description="Binary flag for 3+ month payment delay",
        data_type="binary",
        domain_range="[0, 1]",
        audit_justification="Threshold-based risk flag for severe delinquency",
        is_derived=True,
        derivation_formula="1 if max_delay >= 3 else 0"
    ),
    FeatureDefinition(
        name="payment_ratio",
        category=FeatureCategory.DERIVED,
        description="Ratio of recent payment to recent bill",
        data_type="continuous",
        domain_range="[0, inf)",
        audit_justification="Shows payment adequacy relative to obligation",
        is_derived=True,
        derivation_formula="PAY_AMT1 / (BILL_AMT1 + 1)"  # +1 to avoid division by zero
    ),
    FeatureDefinition(
        name="total_bill",
        category=FeatureCategory.DERIVED,
        description="Sum of all bill amounts over 6 months",
        data_type="continuous",
        domain_range="any",
        audit_justification="Overall credit usage volume",
        is_derived=True,
        derivation_formula="sum(BILL_AMT1, ..., BILL_AMT6)"
    ),
    FeatureDefinition(
        name="total_payment",
        category=FeatureCategory.DERIVED,
        description="Sum of all payment amounts over 6 months",
        data_type="continuous",
        domain_range=">= 0",
        audit_justification="Overall repayment volume",
        is_derived=True,
        derivation_formula="sum(PAY_AMT1, ..., PAY_AMT6)"
    ),
    FeatureDefinition(
        name="payment_consistency",
        category=FeatureCategory.DERIVED,
        description="Standard deviation of monthly payments (lower = more consistent)",
        data_type="continuous",
        domain_range=">= 0",
        audit_justification="Payment consistency indicates financial stability",
        is_derived=True,
        derivation_formula="std(PAY_AMT1, ..., PAY_AMT6)"
    ),
]


# =============================================================================
# ALL FEATURES
# =============================================================================

ALL_FEATURES = (
    DEMOGRAPHIC_FEATURES +
    PAYMENT_HISTORY_FEATURES +
    BILL_AMOUNT_FEATURES +
    PAYMENT_AMOUNT_FEATURES +
    DERIVED_FEATURES
)

ORIGINAL_FEATURE_NAMES = [f.name for f in ALL_FEATURES if not f.is_derived]
DERIVED_FEATURE_NAMES = [f.name for f in ALL_FEATURES if f.is_derived]


def get_feature_by_name(name: str) -> Optional[FeatureDefinition]:
    """Look up feature definition by name."""
    for feature in ALL_FEATURES:
        if feature.name == name:
            return feature
    return None


def get_features_by_category(category: FeatureCategory) -> list[FeatureDefinition]:
    """Get all features in a category."""
    return [f for f in ALL_FEATURES if f.category == category]


def generate_feature_audit_table() -> str:
    """
    Generate a markdown table of all features for audit documentation.
    """
    lines = [
        "| Feature | Category | Type | Justification |",
        "|---------|----------|------|---------------|"
    ]
    
    for feature in ALL_FEATURES:
        lines.append(
            f"| {feature.name} | {feature.category.value} | "
            f"{feature.data_type} | {feature.audit_justification or 'N/A'} |"
        )
    
    return "\n".join(lines)
