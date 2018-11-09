"""
Unit and regression test for the drug_bandit package.
"""

# Import package, test suite, and other packages as needed
import drug_bandit
import pytest
import sys

def test_drug_bandit_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "drug_bandit" in sys.modules
