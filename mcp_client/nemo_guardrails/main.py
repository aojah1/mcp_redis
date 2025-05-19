import os
# ─── NVIDIA Nemo Guardrails ──────────────────────────────
from nemoguardrails import LLMRails, RailsConfig


def get_file_path(filename):
 script_dir = os.path.dirname(os.path.abspath(__file__))
 return os.path.join(script_dir, filename)


# rails_config = RailsConfig.from_content(
#        colang_content=open(get_file_path('nemo_guardrails/rails.config'), 'r').read(),
#        yaml_content=open(get_file_path('nemo_guardrails/config.yml'), 'r').read()
#    )
# ─── NVIDIA Nemo Guardrails spec ──────────────────────────────
# Refuse any politics-related user input
POLITICS_RAIL = """
version: 1
metadata:
  name: no-politics
inputs:
  user_input: str
outputs:
  response: str
completion:
  instructions:
    - when: user_input.lower() matches /(politics|election|government|vote)/
      response: "I’m sorry, I can’t discuss politics."
    - when: true
      response: "{% do %} {{ user_input }} {% enddo %}"
"""
def rails_config():
 return RailsConfig.from_content(colang_content=POLITICS_RAIL)

