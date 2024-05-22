{{ fullname | smart_fullname | escape | underline}}

.. *Based on custom templaste* ``class.rst``

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :private-members:
   :show-inheritance:
   :inherited-members:

   {% block methods %}
   .. automethod:: __init__
   
   {% if all_methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :nosignatures:
   {% for item in all_methods %}
      {% if not item.startswith('__') %}
         ~{{ name }}.{{ item }}
      {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if all_attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in all_attributes %}
      {% if not item.startswith('__') %}
         ~{{ name }}.{{ item }}
      {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

