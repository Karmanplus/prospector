"""Detached-launch entry: ``python -m prospector.trades.design_search``.

The app starts this as a subprocess and polls the session directory. Not a command-line tool -- see
:mod:`prospector.trades.design_search.args`.
"""
import sys

from prospector.trades.design_search.args import main

if __name__ == "__main__":
    sys.exit(main())
