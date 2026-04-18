{% extends "!autosummary/base.rst" %}

{% block body %}
{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ objname }}
   {% if objtype == "class" %}
   :members:
   :show-inheritance:
   :inherited-members:
   {% elif objtype == "function" %}
   :noindex:
   {% endif %}
{% endblock %}
