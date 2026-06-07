#!/usr/bin/env python3
"""DEPRECATED — use kalshi_mlb_unified.py instead.

The unified script covers ALL MLB market types:
  KXMLBKS → strikeouts
  KXMLBHR → home runs
  KXMLBTB → total bases
  KXMLBHRR → hits+runs+RBIs

Usage:
    python -m src.scripts.kalshi_mlb_unified --scan
    python -m src.scripts.kalshi_mlb_unified --bet
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.scripts.kalshi_mlb_unified import main

if __name__ == "__main__":
    print("NOTE: kalshi_mlb.py is deprecated. Using kalshi_mlb_unified.py.\n")
    main()
