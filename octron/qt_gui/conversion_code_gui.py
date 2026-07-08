# Code to correct the octron.py file
# that is created when running
# uic -g python octron.ui > octron.py

# Rational: The octron.py contains some errors that need to be corrected.
# Some of the changes that need to be made are:
# - Replace PySide2 imports with qtpy imports
# - Replace "octron_widgetui" with "self" when used as a parameter
# - Add "qt_gui/" in front of every SVG path

import re

# Define the file paths
input_file_path = "octron.py"
output_file_path = "octron_corrected.py"

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
    r"def setupUi\(self, octron_widgetui\)", "def setupUi(self)", content
)

# Add the parameter "base_path" to the setupUi method definition
content = re.sub(
    r"def setupUi\(self\):", "def setupUi(self, base_path):", content
)

# Replace "octron_widgetui" with "self" when used as a parameter
content = re.sub(r"\boctron_widgetui\b", "self", content)

# Add base_path in front of 'qt_gui/' within every SVG path in the strings,
# with the variable appearing as {base_path} using an f-string with double braces.
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

# Change 'self.retranslateUi(octron_widgetui)' to 'self.retranslateUi()'
content = re.sub(
    r"self.retranslateUi\(self\)", "self.retranslateUi()", content
)

# Delete the self.retranslateUi() call entirely
content = re.sub(r"\s*self\.retranslateUi\(\)\n", "\n", content)
# Delete only the retranslateUi method header line, leaving its body intact.
content = re.sub(
    r"\n\s*def retranslateUi\(self\):\n", "\n", content, flags=re.MULTILINE
)


# Convert every "self." underneath the setupUi header to "self.octron." for the entire function body.
# This pattern captures the function header and the complete body until the next top-level definition or EOF.
content = re.sub(
    r"(def setupUi\(self, base_path\):\n)([\s\S]+?)(?=^def |\Z)",
    lambda m: (
        m.group(1)
        + re.sub(r"\bself\.(?!octron\.)", "self.octron.", m.group(2))
    ),
    content,
    flags=re.MULTILINE,
)

# Replace the video_file_drop_widget assignment line with the desired replacement.
content = re.sub(
    r"self\.octron\.video_file_drop_widget\s*=\s*QWidget\(self\.octron\.project_video_drop_groupbox\)",
    "self.octron.video_file_drop_widget = Mp4DropWidget()",
    content,
)

# Replace the video_file_drop_widget assignment for YOLO predict line with the desired replacement.
content = re.sub(
    r"self\.octron\.predict_video_drop_widget\s*=\s*QWidget\(self\.octron\.predict_video_drop_groupbox\)",
    "self.octron.predict_video_drop_widget = Mp4DropWidget()",
    content,
)


# Write the corrected content to the output file
with open(output_file_path, "w") as file:
    file.write(content)

print(f"Corrected code has been written to {output_file_path}")
