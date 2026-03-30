from typing import List

from cldk.models.python import PyFunction

from tangent.agent_analysis.model.models import CallableDetails


class CommonAnalysis:
    def __init__(self, analysis):
        self.analysis = analysis

    def extract_application_calls(self, function: PyFunction)->List[CallableDetails]:
        pass

    def extract_constructor_calls(self, function: PyFunction)->List[CallableDetails]:
        pass

    def extract_library_calls(self, function: PyFunction)->List[CallableDetails]:
        pass