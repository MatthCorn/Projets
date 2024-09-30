import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GradObserver.App import RunAnalyser

RunAnalyser(None)