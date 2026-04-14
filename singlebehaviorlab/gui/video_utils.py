"""Utility functions for video file handling."""
import os
import shutil
from typing import Optional, List
from PyQt6.QtWidgets import QMessageBox, QWidget


def ensure_videos_in_experiment(
    video_paths: List[str],
    config: dict,
    parent_widget: Optional[QWidget] = None
) -> List[str]:
    """
    Ensure multiple videos are in experiment's raw_videos folder.
    
    For multiple videos, asks once if user wants to copy all of them.
    
    Args:
        video_paths: List of paths to video files
        config: Application config dictionary
        parent_widget: Optional parent widget for showing dialogs
        
    Returns:
        List of paths to video files (original or copied)
    """
    if not video_paths:
        return []
    
    if len(video_paths) == 1:
        return [ensure_video_in_experiment(video_paths[0], config, parent_widget)]
    
    # Multiple videos - check which ones need copying
    experiment_path = config.get("experiment_path")
    if not experiment_path or not os.path.exists(experiment_path):
        return video_paths
    
    raw_videos_dir = config.get("raw_videos_dir")
    if not raw_videos_dir:
        raw_videos_dir = os.path.join(experiment_path, "data", "raw_videos")
    
    videos_to_copy = []
    videos_already_there = []
    videos_with_conflicts = []
    
    raw_videos_abs_dir = os.path.abspath(raw_videos_dir)
    
    for video_path in video_paths:
        if not os.path.exists(video_path):
            continue
        
        video_abs_path = os.path.abspath(video_path)
        
        if video_abs_path.startswith(raw_videos_abs_dir):
            videos_already_there.append(video_path)
        else:
            video_name = os.path.basename(video_path)
            target_path = os.path.join(raw_videos_dir, video_name)
            
            if os.path.exists(target_path) and not os.path.samefile(video_path, target_path):
                videos_with_conflicts.append((video_path, target_path))
            else:
                videos_to_copy.append((video_path, target_path))
    
    # If all videos are already in experiment folder, return as-is
    if not videos_to_copy and not videos_with_conflicts:
        return video_paths
    
    # Ask user about copying
    if videos_to_copy or videos_with_conflicts:
        msg_parts = []
        if videos_to_copy:
            msg_parts.append(f"{len(videos_to_copy)} video(s) need to be copied to the experiment folder.")
        if videos_with_conflicts:
            msg_parts.append(f"{len(videos_with_conflicts)} video(s) have naming conflicts.")
        
        msg = "\n".join(msg_parts)
        msg += "\n\nWould you like to copy them to the experiment's raw_videos folder?\n"
        msg += "(The original files will remain unchanged)"
        
        if videos_with_conflicts:
            msg += "\n\nNote: Some videos have naming conflicts and will be skipped."
        
        reply = QMessageBox.question(
            parent_widget,
            "Copy Videos to Experiment",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Copy videos
            os.makedirs(raw_videos_dir, exist_ok=True)
            copied_count = 0
            failed_videos = []
            
            for video_path, target_path in videos_to_copy:
                try:
                    shutil.copy2(video_path, target_path)
                    copied_count += 1
                except Exception as e:
                    failed_videos.append((os.path.basename(video_path), str(e)))
            
            # Build result list
            result = []
            for video_path in video_paths:
                if video_path in videos_already_there:
                    result.append(video_path)
                else:
                    video_name = os.path.basename(video_path)
                    target_path = os.path.join(raw_videos_dir, video_name)
                    if os.path.exists(target_path):
                        result.append(target_path)
                    else:
                        result.append(video_path)
            
            if parent_widget:
                if failed_videos:
                    QMessageBox.warning(
                        parent_widget,
                        "Some Videos Failed to Copy",
                        f"Copied {copied_count} video(s) successfully.\n\n"
                        f"Failed to copy {len(failed_videos)} video(s)."
                    )
                else:
                    QMessageBox.information(
                        parent_widget,
                        "Videos Copied",
                        f"Successfully copied {copied_count} video(s) to:\n{raw_videos_dir}"
                    )
            
            return result
    
    return video_paths


def ensure_video_in_experiment(
    video_path: str,
    config: dict,
    parent_widget: Optional[QWidget] = None
) -> str:
    """
    Ensure video is in experiment's raw_videos folder.
    
    If video is not already in the experiment folder, asks user if they want to copy it there.
    Returns the path to use (either original or copied).
    
    Args:
        video_path: Path to the selected video file
        config: Application config dictionary (must contain 'experiment_path' and 'raw_videos_dir')
        parent_widget: Optional parent widget for showing dialogs
        
    Returns:
        Path to video file (original or copied)
    """
    if not os.path.exists(video_path):
        return video_path
    
    experiment_path = config.get("experiment_path")
    if not experiment_path or not os.path.exists(experiment_path):
        # No experiment folder, use original path
        return video_path
    
    raw_videos_dir = config.get("raw_videos_dir")
    if not raw_videos_dir:
        raw_videos_dir = os.path.join(experiment_path, "data", "raw_videos")

    video_abs_path = os.path.abspath(video_path)
    raw_videos_abs_dir = os.path.abspath(raw_videos_dir)

    if video_abs_path.startswith(raw_videos_abs_dir):
        return video_path

    video_name = os.path.basename(video_path)
    target_path = os.path.join(raw_videos_dir, video_name)

    if os.path.exists(target_path):
        if os.path.samefile(video_path, target_path):
            return video_path
        else:
            # Different file with same name - ask user
            reply = QMessageBox.question(
                parent_widget,
                "Video Already Exists",
                f"A video with the name '{video_name}' already exists in the experiment folder.\n\n"
                f"Original: {video_path}\n"
                f"Existing: {target_path}\n\n"
                "Do you want to use the existing file in the experiment folder?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            if reply == QMessageBox.StandardButton.Yes:
                return target_path
            else:
                return video_path
    
    # Ask user if they want to copy the video
    reply = QMessageBox.question(
        parent_widget,
        "Copy Video to Experiment",
        f"The selected video is not in the experiment folder.\n\n"
        f"Video: {video_path}\n"
        f"Experiment folder: {experiment_path}\n\n"
        "Would you like to copy it to the experiment's raw_videos folder?\n"
        "(The original file will remain unchanged)",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.Yes
    )
    
    if reply == QMessageBox.StandardButton.Yes:
        # Copy video to experiment folder
        os.makedirs(raw_videos_dir, exist_ok=True)
        try:
            shutil.copy2(video_path, target_path)
            if parent_widget:
                QMessageBox.information(
                    parent_widget,
                    "Video Copied",
                    f"Video copied successfully to:\n{target_path}"
                )
            return target_path
        except Exception as e:
            if parent_widget:
                QMessageBox.warning(
                    parent_widget,
                    "Copy Failed",
                    f"Failed to copy video:\n{str(e)}\n\nUsing original location."
                )
            return video_path
    else:
        # User chose not to copy, use original path
        return video_path

