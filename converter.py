#!/usr/bin/env python3
"""
Video file converter script that lists video files and their metadata.
Uses ffmpeg/ffprobe to extract media stream information.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any


# Common video file extensions
VIDEO_EXTENSIONS = {
    '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm',
    '.m4v', '.mpg', '.mpeg', '.3gp', '.3g2', '.mts', '.m2ts',
    '.vob', '.ogv', '.gifv', '.qt', '.f4v', '.asf', '.rm',
    '.rmvb', '.divx', '.xvid'
}


def is_video_file(file_path: Path) -> bool:
    """Check if a file is a video file based on its extension."""
    return file_path.suffix.lower() in VIDEO_EXTENSIONS


def get_video_files(directory: Path, recursive: bool = False) -> List[Path]:
    """
    Get all video files in a directory.
    
    Args:
        directory: Path to the directory to scan
        recursive: If True, scan subdirectories recursively
    
    Returns:
        List of Path objects for video files
    """
    video_files = []
    
    if recursive:
        # Use rglob for recursive search
        for file_path in directory.rglob('*'):
            if file_path.is_file() and is_video_file(file_path):
                video_files.append(file_path)
    else:
        # Use glob for non-recursive search
        for file_path in directory.glob('*'):
            if file_path.is_file() and is_video_file(file_path):
                video_files.append(file_path)
    
    return sorted(video_files)


def get_file_size(file_path: Path) -> int:
    """Get file size in bytes."""
    return file_path.stat().st_size


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def get_video_info(file_path: Path) -> Dict[str, Any]:
    """
    Get video file information using ffprobe.
    
    Args:
        file_path: Path to the video file
    
    Returns:
        Dictionary containing file information and stream details
    """
    info = {
        'filename': file_path.name,
        'path': str(file_path),
        'size': get_file_size(file_path),
        'size_formatted': format_file_size(get_file_size(file_path)),
        'streams': [],
        'error': None
    }
    
    try:
        # Use ffprobe to get video information in JSON format
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            str(file_path)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            
            # Extract stream information
            for stream in data.get('streams', []):
                stream_info = {
                    'type': stream.get('codec_type', 'unknown'),
                    'codec': stream.get('codec_name', 'unknown'),
                }
                
                # Add additional info based on stream type
                if stream.get('codec_type') == 'video':
                    stream_info['width'] = stream.get('width')
                    stream_info['height'] = stream.get('height')
                    stream_info['fps'] = stream.get('r_frame_rate', 'unknown')
                elif stream.get('codec_type') == 'audio':
                    stream_info['sample_rate'] = stream.get('sample_rate')
                    stream_info['channels'] = stream.get('channels')
                
                info['streams'].append(stream_info)
        else:
            info['error'] = f"ffprobe failed: {result.stderr}"
    
    except subprocess.TimeoutExpired:
        info['error'] = "ffprobe timeout"
    except FileNotFoundError:
        info['error'] = "ffprobe not found - please install ffmpeg"
    except json.JSONDecodeError:
        info['error'] = "Failed to parse ffprobe output"
    except Exception as e:
        info['error'] = f"Error: {str(e)}"
    
    return info


def format_stream_info(stream: Dict[str, Any]) -> str:
    """Format stream information as a string."""
    parts = [f"Type: {stream['type']}", f"Codec: {stream['codec']}"]
    
    if stream['type'] == 'video' and stream.get('width') and stream.get('height'):
        parts.append(f"Resolution: {stream['width']}x{stream['height']}")
        if stream.get('fps'):
            parts.append(f"FPS: {stream['fps']}")
    elif stream['type'] == 'audio':
        if stream.get('channels'):
            parts.append(f"Channels: {stream['channels']}")
        if stream.get('sample_rate'):
            parts.append(f"Sample Rate: {stream['sample_rate']} Hz")
    
    return ', '.join(parts)


def print_video_list(video_files: List[Path], directory: Path):
    """
    Print information about video files.
    
    Args:
        video_files: List of video file paths
        directory: Base directory being scanned
    """
    if not video_files:
        print("No video files found.")
        return
    
    print(f"\nFound {len(video_files)} video file(s):\n")
    print("=" * 80)
    
    for idx, file_path in enumerate(video_files, 1):
        print(f"\n{idx}. {file_path.name}")
        print(f"   Path: {file_path.relative_to(directory) if file_path.is_relative_to(directory) else file_path}")
        
        info = get_video_info(file_path)
        
        print(f"   Size: {info['size_formatted']} ({info['size']:,} bytes)")
        
        if info['error']:
            print(f"   Error: {info['error']}")
        elif info['streams']:
            print(f"   Streams: {len(info['streams'])}")
            for stream_idx, stream in enumerate(info['streams'], 1):
                print(f"      Stream {stream_idx}: {format_stream_info(stream)}")
        else:
            print("   No stream information available")
    
    print("\n" + "=" * 80)


def main():
    """Main function to parse arguments and list video files."""
    parser = argparse.ArgumentParser(
        description='List video files in a directory and display their metadata using ffmpeg.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 converter.py /path/to/videos
  python3 converter.py -r /path/to/videos
        """
    )
    
    parser.add_argument(
        '-r',
        '--recursive',
        action='store_true',
        help='Scan directories recursively'
    )
    
    parser.add_argument(
        'directory',
        type=str,
        help='Directory to scan for video files'
    )
    
    args = parser.parse_args()
    
    # Convert directory to Path object and validate
    directory = Path(args.directory)
    
    if not directory.exists():
        print(f"Error: Directory '{args.directory}' does not exist.", file=sys.stderr)
        sys.exit(1)
    
    if not directory.is_dir():
        print(f"Error: '{args.directory}' is not a directory.", file=sys.stderr)
        sys.exit(1)
    
    # Get video files
    mode = "recursive" if args.recursive else "non-recursive"
    print(f"Scanning directory '{directory}' ({mode} mode)...")
    
    video_files = get_video_files(directory, args.recursive)
    
    # Print results
    print_video_list(video_files, directory)


if __name__ == '__main__':
    main()
