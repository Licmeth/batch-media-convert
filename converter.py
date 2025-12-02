#!/usr/bin/env python3
"""
Video file listing script that displays video files and their metadata.
Uses ffmpeg/ffprobe to extract media stream information.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set, Optional


# Common video file extensions
VIDEO_EXTENSIONS = {
    '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm',
    '.m4v', '.mpg', '.mpeg', '.3gp', '.3g2', '.mts', '.m2ts',
    '.vob', '.ogv', '.gifv', '.qt', '.f4v', '.asf', '.rm',
    '.rmvb', '.divx', '.xvid'
}


def is_video_file(file_path: Path, extensions: Set[str]) -> bool:
    """Check if a file is a video file based on its extension.
    
    Args:
        file_path: Path to the file to check
        extensions: Set of video extensions to match (e.g., {'.mp4', '.avi'})
    
    Returns:
        True if the file extension matches one of the provided extensions
    """
    ext = file_path.suffix.lower()
    return ext in extensions


def get_video_files(
    directory: Path, 
    recursive: bool = False, 
    extensions: Set[str] = VIDEO_EXTENSIONS
) -> List[Path]:
    """
    Get all video files in a directory.
    
    Args:
        directory: Path to the directory to scan
        recursive: If True, scan subdirectories recursively
        extensions: Set of video extensions to match (defaults to VIDEO_EXTENSIONS)
    
    Returns:
        List of Path objects for video files
    """
    video_files = []
    
    if recursive:
        # Use rglob for recursive search
        for file_path in directory.rglob('*'):
            if file_path.is_file() and is_video_file(file_path, extensions):
                video_files.append(file_path)
    else:
        # Use glob for non-recursive search
        for file_path in directory.glob('*'):
            if file_path.is_file() and is_video_file(file_path, extensions):
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
    file_size = get_file_size(file_path)
    info = {
        'filename': file_path.name,
        'path': str(file_path),
        'size': file_size,
        'size_formatted': format_file_size(file_size),
        'streams': [],
        'duration': None,
        'bytes_per_sec_per_pixel': None,
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
            first_video_stream = None
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
                    stream_info['duration'] = stream.get('duration')
                    
                    # Store first video stream for bytes per second per pixel calculation
                    if first_video_stream is None:
                        first_video_stream = stream
                elif stream.get('codec_type') == 'audio':
                    stream_info['sample_rate'] = stream.get('sample_rate')
                    stream_info['channels'] = stream.get('channels')
                
                info['streams'].append(stream_info)
            
            # Calculate bytes per second per pixel for first video stream
            if first_video_stream:
                duration = first_video_stream.get('duration')
                width = first_video_stream.get('width')
                height = first_video_stream.get('height')
                
                if duration and width and height:
                    try:
                        duration_float = float(duration)
                        total_pixels = width * height
                        if duration_float > 0 and total_pixels > 0:
                            bytes_per_sec = file_size / duration_float
                            info['bytes_per_sec_per_pixel'] = bytes_per_sec / total_pixels
                            info['duration'] = duration_float
                    except (ValueError, ZeroDivisionError):
                        pass
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


def sort_video_data(
    video_data: List[Tuple[Path, Dict[str, Any]]], 
    sort_by: str = 'path', 
    reverse: bool = False
) -> List[Tuple[Path, Dict[str, Any]]]:
    """
    Sort video data by the specified metric.
    
    Args:
        video_data: List of tuples containing (file_path, video_info)
        sort_by: Metric to sort by ('path', 'size', or 'bps_per_pixel')
        reverse: If True, reverse the sort order
    
    Returns:
        Sorted list of tuples containing (file_path, video_info)
    """
    if sort_by == 'size':
        video_data.sort(key=lambda x: x[1]['size'], reverse=reverse)
    elif sort_by == 'bps_per_pixel':
        # Files without bytes_per_sec_per_pixel will be sorted to the end
        video_data.sort(
            key=lambda x: (x[1]['bytes_per_sec_per_pixel'] is None, 
                          x[1]['bytes_per_sec_per_pixel'] or 0),
            reverse=reverse
        )
    else:  # sort by path (default)
        video_data.sort(key=lambda x: str(x[0]), reverse=reverse)
    
    return video_data


def print_video_entry(
    file_path: Path, 
    info: Dict[str, Any], 
    directory: Path, 
    idx: int
) -> None:
    """
    Print a single video file entry with its metadata.
    
    Args:
        file_path: Path to the video file
        info: Video information dictionary
        directory: Base directory being scanned
        idx: Index number for display
    """
    print(f"\n{idx}. {file_path.name}")
    # Try to show relative path, fallback to absolute path
    try:
        rel_path = file_path.relative_to(directory)
        print(f"   Path: {rel_path}")
    except ValueError:
        print(f"   Path: {file_path}")
    
    print(f"   Size: {info['size_formatted']} ({info['size']:,} bytes)")
    
    if info['error']:
        print(f"   Error: {info['error']}")
    elif info['streams']:
        # Display bytes per second per pixel if available
        if info['bytes_per_sec_per_pixel'] is not None:
            print(f"   Bytes/sec/pixel: {info['bytes_per_sec_per_pixel']:.6f}")
        elif info['duration'] is None:
            print(f"   Bytes/sec/pixel: N/A (duration not available)")
        else:
            print(f"   Bytes/sec/pixel: N/A")
        
        print(f"   Streams: {len(info['streams'])}")
        for stream_idx, stream in enumerate(info['streams'], 1):
            print(f"      Stream {stream_idx}: {format_stream_info(stream)}")
    else:
        print("   No stream information available")


def print_video_list(
    video_files: List[Path], 
    directory: Path, 
    sort_by: str = 'path', 
    reverse: bool = False
) -> None:
    """
    Print information about video files.
    
    Args:
        video_files: List of video file paths
        directory: Base directory being scanned
        sort_by: Metric to sort by ('path', 'size', or 'bps_per_pixel')
        reverse: If True, reverse the sort order
    """
    if not video_files:
        print("No video files found.")
        return
    
    print(f"\nFound {len(video_files)} video file(s), sorted by {sort_by}{' (reversed)' if reverse else ''}:\n")
    print("=" * 80)
    
    # Get video info for all files first (needed for sorting)
    video_data: List[Tuple[Path, Dict[str, Any]]] = []
    for file_path in video_files:
        info = get_video_info(file_path)
        video_data.append((file_path, info))
    
    # Sort the video data
    video_data = sort_video_data(video_data, sort_by, reverse)
    
    # Print each video entry
    for idx, (file_path, info) in enumerate(video_data, 1):
        print_video_entry(file_path, info, directory, idx)
    
    print("\n" + "=" * 80)


def setup_and_parse_arguemts() -> argparse.Namespace:
    """Set up the argument parser for the script and return the parsed arguments."""
    parser = argparse.ArgumentParser(
        description='List video files in a directory and display their metadata using ffmpeg.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 converter.py /path/to/videos
  python3 converter.py /path/to/videos -r
  python3 converter.py /path/to/videos -s size --reverse
  python3 converter.py /path/to/videos -i .mp4,.avi
  python3 converter.py /path/to/videos -i mp4,mkv,avi -r
  python3 converter.py /path/to/videos -e .wmv,.flv
  python3 converter.py /path/to/videos -i mp4,mkv -e mp4 -s bps_per_pixel
        """
    )

    parser.add_argument(
        '-r',
        '--recursive',
        action='store_true',
        help='Scan directories recursively'
    )
    
    parser.add_argument(
        '-s',
        '--sort-by',
        choices=['path', 'size', 'bps_per_pixel'],
        default='path',
        help='Sort output by: path (default), size, or bps_per_pixel (bytes per second per pixel)'
    )
    
    parser.add_argument(
        '--reverse',
        action='store_true',
        help='Reverse the sort order'
    )
    
    parser.add_argument(
        '-i',
        '--include-ext',
        type=str,
        metavar='EXT1,EXT2,...',
        help='Include only specific video extensions as comma-separated list (e.g., -i .mp4,.avi or -i mp4,avi).'
    )

    parser.add_argument(
        '-e',
        '--exclude-ext',
        type=str,
        metavar='EXT1,EXT2,...',
        help='Exclude specific video extensions as comma-separated list (e.g., -e .mp4,.avi or -e mp4,avi).'
    )
    
    parser.add_argument(
        'directory',
        type=str,
        help='Directory to scan for video files'
    )

    return parser.parse_args()


def main():
    """Main function to parse arguments and list video files."""
    args = setup_and_parse_arguemts()

    # Process extensions
    extensions: Set[str] = VIDEO_EXTENSIONS

    # Use only specific extensions if provided
    if args.include_ext:
        # Normalize extensions to lowercase and ensure they start with a dot
        extensions = set()
        for ext in args.include_ext.split(','):
            ext = ext.strip().lower()
            if not ext:
                continue
            if not ext.startswith('.'):
                ext = '.' + ext
            extensions.add(ext)
    
    # Exclude specific extensions if provided
    if args.exclude_ext:
        for ext in args.exclude_ext.split(','):
            ext = ext.strip().lower()
            if not ext:
                continue
            if not ext.startswith('.'):
                ext = '.' + ext
            extensions.discard(ext)
    
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
    if args.include_ext:
        print(f"Including only extensions: {', '.join(sorted(extensions))}")
    if args.exclude_ext:
        excluded_exts = [e.strip() for e in args.exclude_ext.split(',') if e.strip()]
        print(f"Excluded extensions: {', '.join(excluded_exts)}")
    
    video_files = get_video_files(directory, args.recursive, extensions)
    
    # Print results
    print_video_list(video_files, directory, sort_by=args.sort_by, reverse=args.reverse)


if __name__ == '__main__':
    main()
