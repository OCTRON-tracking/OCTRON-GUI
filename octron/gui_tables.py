# QT Table views
# This is used in OCTRON for example in the project data view to display the label data in a table view
from pathlib import Path        
from qtpy.QtCore import Qt, QAbstractTableModel, Signal
#from qtpy.QtGui import QFont # Skip for now since font is unchanged (see commented )

class ExistingDataTable(QAbstractTableModel):
    """
    
    Table model for displaying label data that 
    has been found in .json organizer files
    via the collect_labels() function.
      
    The goal is to present the user with a table view 
    of already existing labeled data throughout a project
    folder.
    
    
    
    """
    # Add a signal for double-click events
    doubleClicked = Signal(str)
    
    
    def __init__(self, label_dict=None):
        super().__init__()
        self.label_dict = label_dict or {}
        self.headers = ["Folder name", "Video name", "# Labels", "# Frames"]
        self._data = []
        self.refresh_data()
    
    def refresh_data(self):
        """Process the label_dict into table rows"""
        self._data = []
        
        # No data case
        if not self.label_dict:
            return
        # Process each video folder entry
        for folder_name, labels in self.label_dict.items():
                
            total_labels = 0
            unique_frames = set()  # Use a set to track unique frame indices
            
            # before looping labels, init tooltip holder
            full_video_hint = ""
            label_names = [] 
            # Count labels and frames
            for label_id, label_data in labels.items():
                if label_id == 'video':
                    continue
                if label_id == 'video_file_path':
                    p = Path(label_data)  
                    full_video_hint = "/".join(p.parts[-3:])
                    video_file_path = p.stem[-7:]
                    continue 
                total_labels += 1
                label_names.append(label_data['label']) 
                # Add frame indices to the set of unique frames
                if 'frames' in label_data:
                    unique_frames.update(label_data['frames'])
            
            label_names.sort()
            # Create a row for this folder
            shortened_folder_name = Path(folder_name).name
            self._data.append(
                             [shortened_folder_name, 
                              video_file_path,
                              total_labels, 
                              len(unique_frames),
                              folder_name,  # folder_name is hidden in display
                              full_video_hint,  # updated: truncated tooltip
                              label_names] 
                              )
    
    def update_data(self, new_label_dict, delete_old=False):
        """Update the model with new data"""
        self.beginResetModel()
        if delete_old:
            # Replace existing data with new data
            self.label_dict = new_label_dict
        else:
            # Merge new data with existing data
            self.label_dict = {**self.label_dict, **new_label_dict}
        self.refresh_data()
        self.endResetModel()
    
    def rowCount(self, parent=None):
        return len(self._data)
    
    def columnCount(self, parent=None):
        return len(self.headers)
    
    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self._data)):
            return None
            
        row = index.row()
        col = index.column()
        
        if role == Qt.DisplayRole:
            return str(self._data[row][col])
            
        # elif role == Qt.FontRole and col == 0:
        #     # Option to make column bold
        #     #font = QFont()
        #     #font.setBold(False)
        #     #return font
            
        elif role == Qt.TextAlignmentRole and col > 0:
            # Center-align numeric columns
            return Qt.AlignCenter
        
        elif role == Qt.ToolTipRole and index.column() == 1:
            return self._data[row][5]
        
        elif role == Qt.ToolTipRole and index.column() == 2:  
            return ", ".join(self._data[row][6])
        
        elif role == Qt.UserRole:
            return self._data[row][3]  # Return full folder path
            
        return None
    
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.headers[section]
        return None
    
    def remove_row(self, row):
        """Remove a row from the model and its corresponding entry from label_dict"""
        if not (0 <= row < len(self._data)):
            return None
        self.beginResetModel()
        folder_path = self._data[row][4]
        del self._data[row]
        if folder_path in self.label_dict:
            del self.label_dict[folder_path]
        self.endResetModel()
        return folder_path

    def get_folder_path(self, index):
        """Return the full folder path for the given index"""
        if not index.isValid() or not (0 <= index.row() < len(self._data)):
            return None
        return self._data[index.row()][4]
