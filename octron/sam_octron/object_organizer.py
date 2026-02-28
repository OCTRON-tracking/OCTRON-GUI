from pathlib import Path
from shutil import rmtree
import json
import datetime
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Optional, Union, Dict, List, Any

from octron.sam_octron.helpers.sam2_colors import (
    create_label_colors,
    sample_maximally_different,
)


class Obj(BaseModel):
    label: str
    suffix: str
    label_id: Optional[int] = None
    color: Optional[list] = None
    prediction_layer: Optional[Any] = None
    annotation_layer: Optional[Any] = None
    
    # Replace the deprecated class Config with ConfigDict
    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        json_encoders={
            # Add custom encoders for non-serializable types
            tuple: lambda v: list(v),  # Convert tuples to lists
        }
    )

    @field_validator("color")
    def check_color_length(cls, v):
        if v is not None and len(v) != 4:
            raise ValueError("Color must be a list of length 4")
        return v
    
            
    def __repr__(self) -> str:
        return (
            f"Obj(label={self.label!r}, suffix={self.suffix!r}, label_id={self.label_id!r}, "
            f"color={self.color!r}, prediction_layer={self.prediction_layer!r}, annotation_layer={self.annotation_layer!r})"
        )


class ObjectOrganizer(BaseModel):
    """
    This module provides pydantic-based classes for organizing and managing tracking entries in OCTRON. 
    It is designed to help maintain a dictionary of object entries (named Obj, see above), 
    defined each by a unique ID,
    that is also the one that SAM2 internally deals with, 
    and representing attributes such as label, a unique label ID, suffix, color. 
    CAVE: label ID and object ID are not the same! Label IDs are 
    unique for each label and are used to assign colors. The object ID is unique for each object and 
    are used by the SAM2 tracking module.
    It also assigns a unique mask  (=prediction) layer and annotation layer to each object.
    This way, all information for objects flowing through OCTRON is saved in one place. 
    
    
    I am using a color picking strategy that is based on the number of labels and suffixes. 
    A colormap is created (see self.all_colors) and then maximally different colors are picked 
    for each label and suffix combination.
    Labels are assigend "slices" of a colormap and then each suffix is assigned a color from that slice.
    You CAN assign a color yourself, but I would recommend using the internal strategy.
    
    
    Example usage:
    >>> object_organizer = ObjectOrganizer()
    >>> object_organizer.add_entry(0, Obj(label='worm', suffix='0'))
    >>> object_organizer.add_entry(1, Obj(label='worm', suffix='1')) # Color is picked automatically 
    
    
    """
    entries: Dict[int, Obj] = {}
    # Mapping from label to its assigned label_id.
    label_id_map: Dict[str, int] = {}
    # The next available label_id.
    next_label_id: int = 0
    
    # Project-level settings (persisted to JSON)
    settings: Dict[str, Any] = {}
    
    # Color dictionary for all labels
    n_labels_max: int = 10 # max number of distinct label colors
    n_subcolors: int  = 50


    def all_colors(self) -> tuple[list, list, list]:
        """
        Create a list of colors for all labels. This is currently capped at 10, 
        which I think is reasonable.
        Returns a list of lists:
            -> label
                -> colors for each label
        """
        label_colors = create_label_colors(cmap='cmr.tropical', 
                                           n_labels=self.n_labels_max, 
                                           n_colors_submap=self.n_subcolors,
                                           )
        # Create maximally different colors for each label and for each submap 
        indices_max_diff_labels = sample_maximally_different(list(range(self.n_labels_max)))
        indices_max_diff_subcolors = sample_maximally_different(list(range(self.n_subcolors)))
        return (label_colors, indices_max_diff_labels, indices_max_diff_subcolors)

    def exists_id(self, id_: int) -> bool:
        return id_ in self.entries
    
    def max_id(self) -> int:
        try:
            return max([e for e in self.entries])
        except ValueError:
            return 0
        
    def min_available_id(self) -> int:
        """
        Find the minimum integer >= 0 that is not yet present as an ID in the object organizer.
        """
        existing_ids = set(self.entries.keys())
        min_id = 0
        while min_id in existing_ids:
            min_id += 1
        return min_id
    
    def exists_label(self, label: str) -> bool:
        return any(entry.label == label for entry in self.entries.values())

    def exists_suffix(self, suffix: str) -> bool:
        return any(entry.suffix == suffix for entry in self.entries.values())

    def exists_combination(self, label: str, suffix: str) -> bool:
        """
        Check if the combination of label and suffix is already present.
        """
        return any(entry.label == label and entry.suffix == suffix for entry in self.entries.values())

    def get_current_labels(self) -> List[str]:
        """
        Return a list of unique labels from all entries.
        """
        return list({entry.label for entry in self.entries.values()})
    
    def get_entries_by_label(self, label: str) -> List[Obj]:
        """
        Return a list of entries (Obj) that have the given label.
        """
        return [entry for entry in self.entries.values() if entry.label == label]

    def get_suffixes_by_label(self, label: str) -> List[str]:
        """
        For a given label, return a list of all suffixes from entries matching that label.
        """
        return [entry.suffix for entry in self.entries.values() if entry.label == label]

    def get_entry_by_label_suffix(self, label: str, suffix: str) -> Optional[Obj]:
        """
        Return the entry that matches the given label and suffix combination.
        """
        for entry in self.entries.values():
            if entry.label == label and entry.suffix == suffix:
                return entry
        return None
    
    def get_annotation_layers(self) -> list:
        """
        Return a list of all annotation layers in the object organizer.
        Returns
        -------
        list
            List of annotation layers (napari Shapes or Points layers)
        """
        layers = []
        for _, obj in self.entries.items():
            if obj.annotation_layer is not None:
                layers.append(obj.annotation_layer)
        return layers

    def get_prediction_layers(self) -> list:
        """
        Return a list of all prediction layers in the object organizer.
        Returns
        -------
        list
            List of prediction layers (napari Labels layers)
        """
        layers = []
        for _, obj in self.entries.items():
            if obj.prediction_layer is not None:
                layers.append(obj.prediction_layer)
        return layers
    
    def add_entry(self, id_: int, entry: Obj) -> bool:
        if id_ in self.entries:
            raise ValueError(f"ID {id_} already exists.")
        if self.exists_combination(entry.label, entry.suffix):
            if entry.annotation_layer is not None:
                print(f"Combination ({entry.label}, {entry.suffix}) already exists.")
                return False
        # Check if label already exists and assign label_id accordingly.
        if entry.label in self.label_id_map:
            entry.label_id = self.label_id_map[entry.label]
        else:
            entry.label_id = self.next_label_id
            self.label_id_map[entry.label] = self.next_label_id
            self.next_label_id += 1

        # Find out which color to assign (if necessary)
        if entry.color is None:
            n_subcolors = len(self.get_suffixes_by_label(entry.label)) # These colors already exist ... 
            label_colors, indices_max_diff_labels, indices_max_diff_subcolors = self.all_colors()

            colors_index = indices_max_diff_labels[entry.label_id % self.n_labels_max]
            subcolors_index = indices_max_diff_subcolors[n_subcolors % self.n_subcolors]
            
            this_color = label_colors[colors_index][subcolors_index]
            entry.color = this_color
            
        self.entries[id_] = entry
        return True
    
    def update_entry(self, id_: int, entry: Obj) -> None:
        if id_ not in self.entries:
            raise ValueError(f"ID {id_} does not exist.")
        for eid, existing in self.entries.items():
            if eid != id_ and existing.label == entry.label and existing.suffix == entry.suffix:
                raise ValueError(f"Combination ({entry.label}, {entry.suffix}) already exists in ID {eid}.")
        self.entries[id_] = entry

    def get_entry_id(self, organizer_entry: Obj) -> Optional[int]:
        """
        Return the ID of the given organizer entry.
        """
        for id_, entry in self.entries.items():
            if entry == organizer_entry:
                return id_
        return None

    def get_entry(self, id_: int) -> Optional[Obj]:
        """
        Return the entry for the given id.
        """
        if id_ not in self.entries:
            return None
        return self.entries[id_]

    def remove_entry(self, id_: int) -> Optional[Obj]:
        """
        Remove the entry for the given id and return it.
        Raises KeyError if id is not found.
        """
        if id_ not in self.entries:
            return None
        return self.entries.pop(id_)
    
    def save_to_disk(self, file_path: Union[str, Path]):
        """
        Save the object organizer to disk as JSON
        """
        file_path = Path(file_path)
        # Create a copy of the data without non-serializable objects
        serializable_data = {
            "entries": {},
            "settings": self.settings,
            "time_last_changed": datetime.datetime.now().isoformat()  # Add current timestamp in ISO format
        }
        if not self.entries:
            print("⚠️ No entries to save. Deleting object organizer file and zarr files.")
            if file_path.exists():
                file_path.unlink()
            for zarr_file in file_path.parent.rglob('*.zarr'):
                rmtree(zarr_file)
            return
        
        for obj_id, obj in self.entries.items():
            # Create a serializable version without the layers
            serializable_obj = obj.model_dump(exclude={"annotation_layer", "prediction_layer"})
            # Then add metadata info back in 
            # Add metadata about the annotation layer
            if obj.annotation_layer is not None:
                serializable_obj["annotation_layer_metadata"] = {
                    "name": obj.annotation_layer.name,
                    "type": obj.annotation_layer._basename(),  # 'Shapes' or 'Points'
                    "visible": obj.annotation_layer.visible,
                    "opacity": obj.annotation_layer.opacity,
                }
            # Add metadata about the prediction (mask, ... ) layer
            if obj.prediction_layer is not None:
                prediction_layer_data = obj.prediction_layer.data
                prediction_layer_meta = obj.prediction_layer.metadata
                predicted_indices = np.where(prediction_layer_data[:,0,0] >= 0)[0]
                if len(predicted_indices):
                    num_predicted_indices = len(predicted_indices)
                else:
                    num_predicted_indices = 0
                serializable_obj["prediction_layer_metadata"] = {
                    "name": obj.prediction_layer.name,
                    "type": obj.prediction_layer._basename(),  # 'Labels'
                    "num_predicted_indices": num_predicted_indices,
                    "data_shape": list(obj.prediction_layer.data.shape) 
                                if hasattr(obj.prediction_layer.data, 'shape') else None,
                    "ndim": obj.prediction_layer.ndim,
                    "visible": obj.prediction_layer.visible,
                    "opacity": obj.prediction_layer.opacity,
                    "zarr_path" : prediction_layer_meta['_zarr'].as_posix(),
                    "video_file_path": prediction_layer_meta['_video_file_path'].as_posix(),
                    "video_hash": prediction_layer_meta['_hash'],
                    
                }
                
                # Handle colormap serialization based on its type
                colormap = obj.prediction_layer.colormap
                if colormap is not None:
                    # For DirectLabelColormap (from napari utils), extract the colors
                    if hasattr(colormap, "name"):
                        serializable_obj["prediction_layer_metadata"]["colormap_name"] = colormap.name
                      
                # Add zarr file path if it exists in metadata
                if hasattr(obj.prediction_layer, 'metadata') and '_zarr' in obj.prediction_layer.metadata:
                    serializable_obj["prediction_layer_metadata"]["zarr_path"] = obj.prediction_layer.metadata['_zarr'].as_posix()
            
            serializable_data["entries"][str(obj_id)] = serializable_obj  # Convert key to string for JSON compatibility
            
        # Write to file
        if file_path.exists():
            print(f"⚠️ Overwriting existing file at {file_path.as_posix()}")
            file_path.unlink()
        with open(file_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        print(f"💾 Octron object organizer saved to {file_path.as_posix()}")
        



    def __repr__(self) -> str:
        output_lines = ["OCTRON ObjectOrganizer with entries:"]
        for id_, entry in self.entries.items():
            output_lines.append(
                f"  ID {id_}: label='{entry.label}', suffix='{entry.suffix}', label_id='{entry.label_id}', color='{entry.color}'"
            )
        return "\n".join(output_lines)
 
 
 