"""Code to correct the octron_prediction_cleaner.py file.

That is created when running
``uic -g python octron_prediction_cleaner.ui > octron_prediction_cleaner.py``.

This mirrors ``conversion_code_gui.py`` (used for the main octron widget),
adapted for the prediction cleaner widget. Some of the changes that need to
be made are:
- Replace PySide2 imports with qtpy imports
- Replace "octron_prediction_cleaner" with "self" when used as a parameter
- Fold retranslateUi() into setupUi() (inlined, not called as a separate
  method), matching how gui_elements.py consumes octron_corrected.py
"""

import re

# Define the file paths
input_file_path = "octron_prediction_cleaner.py"
output_file_path = "octron_prediction_cleaner_corrected.py"

# Read the content of the input file
with open(input_file_path) as file:
    content = file.read()

# Replace PySide2 imports with qtpy imports
content = re.sub(
    r"from PySide2.QtCore import \*", "from qtpy.QtCore import *", content
)
content = re.sub(
    r"from PySide2.QtGui import \*", "from qtpy.QtGui import *", content
)
content = re.sub(
    r"from PySide2.QtWidgets import \*",
    "from qtpy.QtWidgets import *",
    content,
)

# Modify the setupUi method signature
content = re.sub(
    r"def setupUi\(self, octron_prediction_cleaner\)",
    "def setupUi(self)",
    content,
)

# Add the parameter "base_path" to the setupUi method definition
content = re.sub(
    r"def setupUi\(self\):", "def setupUi(self, base_path):", content
)

# Replace "octron_prediction_cleaner" with "self" when used as a parameter.
# Excludes matches immediately followed by ".svg"/".ui" -- the widget's own
# icon file and the source .ui file happen to share this exact basename,
# so a plain \b-bounded substitution would corrupt those filename string
# literals (and the header comment) instead of only replacing the
# parameter's attribute-access usages (octron_prediction_cleaner.<method>).
content = re.sub(
    r"\boctron_prediction_cleaner\b(?!\.(?:svg|ui)\b)", "self", content
)

# Add base_path in front of 'qt_gui/' within every SVG path in the strings,
# with the variable appearing as {base_path} using an f-string with
# double braces.
content = re.sub(
    r'\bu"([^"]*\.svg")',
    lambda m: f'f"{{base_path}}/qt_gui/{m.group(1)}',
    content,
)
# Delete any lines containing "connectSlotsByName"
content = re.sub(
    r"^.*connectSlotsByName.*$\n?", "", content, flags=re.MULTILINE
)

# Remove the second 'self' in retranslateUi method
content = re.sub(
    r"def retranslateUi\(self, self\)", "def retranslateUi(self)", content
)

# Change 'self.retranslateUi(self)' to 'self.retranslateUi()'
content = re.sub(
    r"self.retranslateUi\(self\)", "self.retranslateUi()", content
)

# Delete the self.retranslateUi() call entirely
content = re.sub(r"\s*self\.retranslateUi\(\)\n", "\n", content)
# Delete only the retranslateUi method header line, leaving its body intact.
content = re.sub(
    r"\n\s*def retranslateUi\(self\):\n", "\n", content, flags=re.MULTILINE
)

# Convert every "self." underneath the setupUi header to "self.cleaner."
# for the entire function body. This pattern captures the function
# header and the complete body until the next top-level definition or
# EOF.
content = re.sub(
    r"(def setupUi\(self, base_path\):\n)([\s\S]+?)(?=^def |\Z)",
    lambda m: (
        m.group(1)
        + re.sub(r"\bself\.(?!cleaner\.)", "self.cleaner.", m.group(2))
    ),
    content,
    flags=re.MULTILINE,
)

# Write the corrected content to the output file
with open(output_file_path, "w") as file:
    file.write(content)

print(f"Corrected code has been written to {output_file_path}")
