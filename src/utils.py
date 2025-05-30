"""
Utility functions for data loading, path handling, and preparation.
"""
import os
import pandas as pd
import numpy as np
import logging

from . import config

def load_labels(name_mapping_path):
    """
    Load and process labels from the name mapping CSV.

    Parameters:
    - name_mapping_path: str, path to the CSV file containing subject IDs and grades.

    Returns:
    - data_df: pandas DataFrame with columns [SubjectID, Label].
    """
    try:
        name_mapping_df = pd.read_csv(name_mapping_path)
        print("Name Mapping DataFrame Head:")
        print(name_mapping_df.head())

        # Filter out rows with NA in the specified subject ID column
        valid_mapping_df = name_mapping_df.dropna(subset=[config.SUBJECT_ID_COL])
        print(f"Valid mappings after dropping NA: {len(valid_mapping_df)}")

        # Extract subject IDs and labels
        labels_df = valid_mapping_df[[config.SUBJECT_ID_COL, config.GRADE_COL]].copy()
        print("Labels DataFrame Head:")
        print(labels_df.head())

        # Check for missing grades
        missing_grades = labels_df[labels_df[config.GRADE_COL].isna()]
        if not missing_grades.empty:
            print("\nWarning: Some Subject IDs are missing Grades.")
            print(missing_grades)
            # Drop subjects with missing grades
            labels_df = labels_df.dropna(subset=[config.GRADE_COL])
            print(f"Dropped {len(missing_grades)} subjects with missing Grades.")

        # Map labels to integers using the mapping from config
        labels_df[config.TARGET_LABEL] = labels_df[config.GRADE_COL].map(config.LABEL_MAPPING)

        # Remove any subjects with undefined labels after mapping
        labels_df = labels_df.dropna(subset=[config.TARGET_LABEL])

        # Convert label column to integer
        labels_df[config.TARGET_LABEL] = labels_df[config.TARGET_LABEL].astype(int)

        # Rename subject ID column and keep only SubjectID and Label
        data_df = labels_df[[config.SUBJECT_ID_COL, config.TARGET_LABEL]].rename(
            columns={config.SUBJECT_ID_COL: 'SubjectID'}
        )

        print(f"Loaded {len(data_df)} subjects with valid labels.")
        return data_df

    except FileNotFoundError:
        logging.error(f"Error: The file {name_mapping_path} was not found.")
        raise
    except KeyError as e:
        logging.error(f"Error: Column {e} not found in {name_mapping_path}. Check config.py settings (SUBJECT_ID_COL, GRADE_COL).")
        raise
    except Exception as e:
        logging.error(f"Failed to load and process labels: {e}")
        raise e

def add_image_paths(df, modalities, base_data_dir):
    """
    Adds columns for each modality's file path to the DataFrame.
    Searches for subject folders in base_data_dir directly,
    and then in common subdirectories like 'Training', 'Validation'.
    Assumes files are named {subject_id}_{modality}.nii and located in {data_dir}/{subject_id}/
    """
    df = df.copy()
    possible_subdirs = ["", "Training", "Validation", "Test", "testing", "training", "validation"] # Add more if needed

    for modality in modalities:
        df[modality] = None # Initialize column

    for idx, row in df.iterrows():
        subject_id = row['SubjectID']
        found_path_for_subject = False
        for subdir_name in possible_subdirs:
            prospective_subject_dir = os.path.join(base_data_dir, subdir_name, subject_id)
            if os.path.isdir(prospective_subject_dir):
                # Check if all modality files exist in this directory
                all_modalities_present = True
                temp_paths = {}
                for modality in modalities:
                    file_path = os.path.join(prospective_subject_dir, f"{subject_id}_{modality}.nii")
                    if os.path.isfile(file_path):
                        temp_paths[modality] = file_path
                    else:
                        # Try with .nii.gz as well
                        file_path_gz = os.path.join(prospective_subject_dir, f"{subject_id}_{modality}.nii.gz")
                        if os.path.isfile(file_path_gz):
                            temp_paths[modality] = file_path_gz
                        else:
                            all_modalities_present = False
                            break # A modality is missing in this prospective_subject_dir
                
                if all_modalities_present:
                    for modality in modalities:
                        df.loc[idx, modality] = temp_paths[modality]
                    found_path_for_subject = True
                    break # Found all modalities for this subject in this subdir
        
        if not found_path_for_subject:
            # Fallback to original logic if not found in subdirs, or log a warning
            # Original logic: subject_id folder is directly under base_data_dir
            # This will likely still fail if the above loop didn't find it, but included for robustness.
            subject_dir_direct = os.path.join(base_data_dir, subject_id)
            if os.path.isdir(subject_dir_direct):
                 all_modalities_present_direct = True
                 temp_paths_direct = {}
                 for modality in modalities:
                    file_path = os.path.join(subject_dir_direct, f"{subject_id}_{modality}.nii")
                    if os.path.isfile(file_path):
                        temp_paths_direct[modality] = file_path
                    else:
                        file_path_gz = os.path.join(subject_dir_direct, f"{subject_id}_{modality}.nii.gz")
                        if os.path.isfile(file_path_gz):
                            temp_paths_direct[modality] = file_path_gz
                        else:
                            all_modalities_present_direct = False
                            break
                
                 if all_modalities_present_direct:
                    for modality in modalities:
                        df.loc[idx, modality] = temp_paths_direct[modality]
                 # else:
                 #    logging.warning(f"Could not find all modality files for subject {subject_id} in {subject_dir_direct} or common subdirectories.")
            # else:
            #    logging.warning(f"Subject directory not found for {subject_id} in {base_data_dir} or common subdirectories.")


    # Check if any paths are still None (meaning files weren't found)
    for modality in modalities:
        if df[modality].isnull().any():
            missing_subjects = df[df[modality].isnull()]['SubjectID'].tolist()
            logging.warning(f"Could not construct file path for modality \'{modality}\' for subjects: {missing_subjects[:5]}...") # Show first 5

    return df

def check_file_paths(df, modalities):
    """Checks if all expected image files exist."""
    missing_files = []
    total_checked = 0
    for idx, row in df.iterrows():
        for modality in modalities:
            total_checked += 1
            file_path = row[modality]
            # Check if file_path is None (path not constructed) or if it is not an actual file
            if file_path is None or not os.path.isfile(file_path):
                missing_files.append(str(file_path) if file_path is not None else f"Path not constructed for {row['SubjectID']} - {modality}")

    if missing_files:
        print(f"\nWarning: Found {len(missing_files)} missing files out of {total_checked} checked:")
        # Print only the first few missing files to avoid flooding the console
        for file in missing_files[:10]:
            print(f" - {file}")
        if len(missing_files) > 10:
            print(f"   ... and {len(missing_files) - 10} more.")
        return False
    else:
        print(f"\nChecked {total_checked} file paths. All files are present.")
        return True

def prepare_data_list(df, modalities):
    """
    Prepare a list of data dictionaries for MONAI transforms.
    Each dictionary contains paths for all modalities and the label.
    """
    data_list = []
    required_columns = modalities + [config.TARGET_LABEL, 'SubjectID']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"DataFrame is missing required columns: {missing}. Ensure add_image_paths was called.")

    for idx, row in df.iterrows():
        data_dict = {modality: row[modality] for modality in modalities}
        # MONAI transforms expect label to be part of the dictionary
        data_dict[config.TARGET_LABEL] = np.array(row[config.TARGET_LABEL], dtype=np.float32)
        data_list.append(data_dict)
    return data_list

def print_class_distribution(df, name):
    """Prints the class distribution of a DataFrame."""
    if config.TARGET_LABEL not in df.columns:
        print(f"Warning: Label column '{config.TARGET_LABEL}' not found in {name} DataFrame.")
        return

    class_counts = df[config.TARGET_LABEL].value_counts().sort_index().to_dict()
    total = len(df)
    print(f"\n{name} set: {total} samples")
    if total > 0:
        for label, count in class_counts.items():
            print(f"  Label {label}: {count} samples ({(count/total)*100:.2f}%)")
    else:
        print("  (Empty set)") 