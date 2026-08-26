"""
Unit tests for Payment Processor Module
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src_app.payment import PaymentProcessor

def test_fee_calculation():
    proc = PaymentProcessor(fee_rate=0.02)
    assert proc.calculate_fee(100.0) == 2.0
    assert proc.calculate_fee(50.0) == 1.0

def test_invalid_fee_calculation():
    proc = PaymentProcessor(fee_rate=0.02)
    with pytest.raises(ValueError):
        proc.calculate_fee(-10.0)

def test_charge_processing_valid_card():
    proc = PaymentProcessor()
    assert proc.process_charge("1234567812345678", 50.0) is True

def test_charge_processing_invalid_card():
    proc = PaymentProcessor()
    assert proc.process_charge("1234", 50.0) is False
    assert proc.process_charge("invalidcardnum12", 50.0) is False
