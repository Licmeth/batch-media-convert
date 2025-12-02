#!/usr/bin/env python3
"""
Video file listing script that displays video files and their metadata.
Uses ffmpeg/ffprobe to extract media stream information.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set, Optional

try:
    from send2trash import send2trash
    TRASH_AVAILABLE = True
except ImportError:
    TRASH_AVAILABLE = False


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


def get_and_sort_video_data(
    video_files: List[Path],
    sort_by: str = 'path',
    reverse: bool = False
) -> List[Tuple[Path, Dict[str, Any]]]:
    """
    Get video information for all files and sort them.
    
    Args:
        video_files: List of video file paths
        sort_by: Metric to sort by ('path', 'size', or 'bps_per_pixel')
        reverse: If True, reverse the sort order
    
    Returns:
        List of tuples containing (file_path, video_info), sorted according to parameters
    """
    # Get video info for all files
    video_data: List[Tuple[Path, Dict[str, Any]]] = []
    for file_path in video_files:
        info = get_video_info(file_path)
        video_data.append((file_path, info))
    
    # Sort the video data
    video_data = sort_video_data(video_data, sort_by, reverse)
    
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
    video_data: List[Tuple[Path, Dict[str, Any]]],
    directory: Path,
    sort_by: str = 'path',
    reverse: bool = False
) -> None:
    """
    Print information about video files.
    
    Args:
        video_data: List of tuples containing (file_path, video_info), pre-sorted
        directory: Base directory being scanned
        sort_by: Metric used for sorting (for display purposes)
        reverse: Whether sort order was reversed (for display purposes)
    """
    if not video_data:
        print("No video files found.")
        return
    
    print(f"\nFound {len(video_data)} video file(s), sorted by {sort_by}{' (reversed)' if reverse else ''}:\n")
    print("=" * 80)
    
    # Print each video entry
    for idx, (file_path, info) in enumerate(video_data, 1):
        print_video_entry(file_path, info, directory, idx)
    
    print("\n" + "=" * 80)


def should_convert_file(info: Dict[str, Any], target_codec: str = 'h265') -> Tuple[bool, Optional[str]]:
    """
    Determine if a video file qualifies for conversion to the target codec.
    
    Args:
        info: Video information dictionary from get_video_info()
        target_codec: Target codec ('h264' or 'h265')
    
    Returns:
        Tuple of (should_convert: bool, reason: str or None)
        If should_convert is True, reason is None.
        If should_convert is False, reason explains why.
    """
    if info.get('error'):
        return False, f"File has errors: {info['error']}"
    
    if not info.get('streams'):
        return False, "No stream information available"
    
    # Check if there's at least one video stream
    video_streams = [s for s in info['streams'] if s['type'] == 'video']
    if not video_streams:
        return False, "No video streams found"
    
    # Check if any video stream is already using the target codec
    for stream in video_streams:
        codec = stream.get('codec', '').lower()
        if target_codec == 'h265' and codec in ['hevc', 'h265']:
            return False, f"Already using h265/hevc codec"
        elif target_codec == 'h264' and codec in ['h264', 'avc']:
            return False, f"Already using h264/avc codec"
    
    # File qualifies for conversion
    return True, None


def parse_ffmpeg_progress(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse ffmpeg progress output line.
    
    Args:
        line: Output line from ffmpeg stderr
    
    Returns:
        Dictionary with progress information or None if not a progress line
    """
    # ffmpeg progress lines typically contain time=, frame=, fps=, etc.
    if 'time=' not in line:
        return None
    
    progress = {}
    
    # Extract time (format: HH:MM:SS.MS)
    time_match = re.search(r'time=(\d{2}):(\d{2}):(\d{2}\.\d{2})', line)
    if time_match:
        hours, minutes, seconds = time_match.groups()
        progress['time_seconds'] = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    
    # Extract frame count
    frame_match = re.search(r'frame=\s*(\d+)', line)
    if frame_match:
        progress['frame'] = int(frame_match.group(1))
    
    # Extract fps
    fps_match = re.search(r'fps=\s*([\d.]+)', line)
    if fps_match:
        progress['fps'] = float(fps_match.group(1))
    
    # Extract speed
    speed_match = re.search(r'speed=\s*([\d.]+)x', line)
    if speed_match:
        progress['speed'] = float(speed_match.group(1))
    
    return progress if progress else None


def convert_to_h26x(
    input_path: Path,
    output_path: Path,
    info: Dict[str, Any],
    codec: str,
    preset: str,
    crf: int,
) -> Tuple[bool, str]:
    """
    Convert a video file to h264 or h265 in MKV container with progress reporting.
    
    Args:
        input_path: Path to input video file
        output_path: Path to output MKV file
        info: Video information dictionary (for duration)
        codec: Target codec ('h264' or 'h265')
        preset: Encoding preset (ultrafast to veryslow)
        crf: Constant Rate Factor (1-51, lower is better quality)
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    # Map codec names to ffmpeg library names
    codec_map = {
        'h264': 'libx264',
        'h265': 'libx265'
    }
    ffmpeg_codec = codec_map.get(codec, 'libx265')
    
    # Build ffmpeg command
    # -i: input file
    # -map 0: copy all streams from input
    # -c copy: copy all streams by default
    # -c:v: re-encode video streams to target codec
    # -preset: encoding speed/quality tradeoff
    # -crf: constant rate factor (quality level)
    cmd = [
        'ffmpeg',
        '-i', str(input_path),
        '-map', '0',  # Map all streams from input
        '-c', 'copy',  # Copy streams
        '-c:v', ffmpeg_codec,  # Re-encode video
        '-preset', preset, # Encoding preset
        '-crf', str(crf), # Quality setting
        '-y',  # Overwrite output file if it exists
        str(output_path)
    ]
    
    try:
        print(f"\n   Converting: {input_path.name}")
        print(f"   Output: {output_path.name}")
        print(f"   Command: {' '.join(cmd)}")
        print(f"   Progress: ", end='', flush=True)
        
        # Start ffmpeg process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        duration = info.get('duration')
        last_progress_time = 0
        
        # Read output line by line
        for line in process.stdout:
            progress = parse_ffmpeg_progress(line)
            if progress and 'time_seconds' in progress:
                current_time = progress['time_seconds']
                
                # Update progress every 5 seconds of video time
                if current_time - last_progress_time >= 5:
                    last_progress_time = current_time
                    
                    if duration and duration > 0:
                        percent = min(100, (current_time / duration) * 100)
                        fps = progress.get('fps', 0)
                        speed = progress.get('speed', 0)
                        print(f"\r   Progress: {percent:.1f}% (fps: {fps:.1f}, speed: {speed:.2f}x)", 
                              end='', flush=True)
                    else:
                        fps = progress.get('fps', 0)
                        speed = progress.get('speed', 0)
                        print(f"\r   Progress: {current_time:.1f}s (fps: {fps:.1f}, speed: {speed:.2f}x)", 
                              end='', flush=True)
        
        # Wait for process to complete
        process.wait()
        
        print()  # New line after progress
        
        if process.returncode == 0:
            # Check if output file was created
            if output_path.exists():
                output_size = format_file_size(get_file_size(output_path))
                input_size = format_file_size(info['size'])
                return True, f"Success! Output: {output_size} (Input: {input_size})"
            else:
                return False, "Conversion completed but output file not found"
        else:
            return False, f"FFmpeg exited with code {process.returncode}"
    
    except FileNotFoundError:
        return False, "ffmpeg not found - please install ffmpeg"
    except Exception as e:
        return False, f"Error during conversion: {str(e)}"


def process_conversions(
    video_data: List[Tuple[Path, Dict[str, Any]]],
    output_dir: Optional[Path] = None,
    codec: str = 'h265',
    preset: str = 'slow',
    crf: int = 23
) -> List[Tuple[Path, Dict[str, Any], Dict[str, Any]]]:
    """
    Process conversions for all qualified video files.
    
    Args:
        video_data: List of tuples containing (file_path, video_info)
        output_dir: Directory to save converted files (default: same as input)
        codec: Target codec ('h264' or 'h265')
        preset: Encoding preset (ultrafast to veryslow)
        crf: Constant Rate Factor (1-51)
    
    Returns:
        List of tuples containing (file_path, video_info, conversion_info)
    """
    stats = {
        'total': len(video_data),
        'qualified': 0,
        'skipped': 0,
        'successful': 0,
        'failed': 0
    }

    result = []
    
    print("\n" + "=" * 80)
    print("CONVERSION PHASE")
    print(f"Codec: {codec.upper()}, Preset: {preset}, CRF: {crf}")
    print("=" * 80)
    
    for idx, (file_path, info) in enumerate(video_data, 1):
        print(f"\n{idx}. {file_path.name}")
        
        # Check if file qualifies for conversion
        should_convert, reason = should_convert_file(info, codec)
        
        if not should_convert:
            print(f"   Skipped: {reason}")
            stats['skipped'] += 1
            continue
        
        stats['qualified'] += 1
        print(f"   Qualified for conversion")
        
        # Determine output path
        if output_dir:
            output_path = output_dir / f"{file_path.stem}_{codec}.mkv"
        else:
            output_path = file_path.parent / f"{file_path.stem}_{codec}.mkv"
        
        # Check if output already exists
        if output_path.exists():
            print(f"   Skipped: Output file already exists: {output_path.name}")
            stats['skipped'] += 1
            continue
        
        # Perform conversion
        success, message = convert_to_h26x(file_path, output_path, info, codec, preset, crf)
        
        if success:
            print(f"   {message}")
            stats['successful'] += 1
        else:
            print(f"   Failed: {message}")
            stats['failed'] += 1

        output_info = get_video_info(output_path) if success else {}

        conversion_info = {
            'should_convert': should_convert,
            'reason': reason,
            'status': 'success' if success else 'failed',
            'filename': output_path.name,
            'path': str(output_path),
            'size': output_info.get('size') if success else None,
            'size_formatted': output_info.get('size_formatted') if success else None,
            'streams': output_info.get('streams') if success else None,
            'duration': output_info.get('duration') if success else None,
            'bytes_per_sec_per_pixel': output_info.get('bytes_per_sec_per_pixel') if success else None,
        }

        result.append((file_path, info, conversion_info))
    
    # Print summary
    print("\n" + "=" * 80)
    print("CONVERSION SUMMARY")
    print("=" * 80)
    print(f"Total files: {stats['total']}")
    print(f"Qualified for conversion: {stats['qualified']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Successfully converted: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print("=" * 80)
    
    return result


def delete_file(file_path: Path, additional_info: str) -> bool:
    """Move a file to trash (or delete if trash not available) and return True if successful, False otherwise."""
    try:
        if TRASH_AVAILABLE:
            # Ensure we use absolute path for send2trash (required on Linux)
            absolute_path = file_path.resolve()
            send2trash(str(absolute_path))
            print(f"✓ Moved to trash: {file_path.name} ({additional_info})")
        else:
            file_path.unlink()
            print(f"✓ Deleted file: {file_path.name} ({additional_info})")
            print(f"   Note: Install 'send2trash' to move files to trash instead of permanent deletion")
        return True
    except Exception as e:
        action = "moving to trash" if TRASH_AVAILABLE else "deleting"
        print(f"✗ Error {action} file {file_path.name} ({additional_info}): {e}")
        return False


def review_results(conversion_results: List[Tuple[Path, Dict[str, Any], Dict[str, Any]]]) -> None:
    """
    Review conversion results and allow user to delete original or converted files.
    
    Args:
        conversion_results: List of tuples containing (original_path, original_info, conversion_info)
    """
    # Filter only successful conversions
    successful_conversions = [
        (orig_path, orig_info, conv_info) 
        for orig_path, orig_info, conv_info in conversion_results 
        if conv_info.get('status') == 'success'
    ]
    
    if not successful_conversions:
        print("\nNo successful conversions to review.")
        return
    
    print("\n" + "=" * 80)
    print("CONVERSION REVIEW")
    print("=" * 80)
    print(f"\nReviewing {len(successful_conversions)} successfully converted file(s).\n")

    user_choice = None  # None, 'keep_original', 'keep_converted', 'keep_smaller', 'keep_both'
    bytes_saved_total = 0
    
    for idx, (original_path, original_info, conversion_info) in enumerate(successful_conversions, 1):
        converted_path = Path(conversion_info['path'])
        
        print("=" * 80)
        print(f"\n[{idx}/{len(successful_conversions)}] Conversion Review\n")
        
        # Original file information
        print(f"ORIGINAL FILE: {original_path.name}")
        print(f"  Path: {original_path}")
        print(f"  Size: {original_info['size_formatted']} ({original_info['size']:,} bytes)")
        
        if original_info.get('bytes_per_sec_per_pixel') is not None:
            print(f"  Bytes/sec/pixel: {original_info['bytes_per_sec_per_pixel']:.6f}")
        else:
            print(f"  Bytes/sec/pixel: N/A")
        
        print(f"  Streams: {len(original_info['streams'])}")
        for stream_idx, stream in enumerate(original_info['streams'], 1):
            print(f"    Stream {stream_idx}: {format_stream_info(stream)}")
        
        # Converted file information
        print(f"\nCONVERTED FILE: {conversion_info['filename']}")
        print(f"  Path: {converted_path}")
        print(f"  Size: {conversion_info['size_formatted']} ({conversion_info['size']:,} bytes)")
        
        if conversion_info.get('bytes_per_sec_per_pixel') is not None:
            print(f"  Bytes/sec/pixel: {conversion_info['bytes_per_sec_per_pixel']:.6f}")
        else:
            print(f"  Bytes/sec/pixel: N/A")
        
        if conversion_info['streams']:
            print(f"  Streams: {len(conversion_info['streams'])}")
            for stream_idx, stream in enumerate(conversion_info['streams'], 1):
                print(f"    Stream {stream_idx}: {format_stream_info(stream)}")
        
        # Comparison
        print("\nCOMPARISON:")
        size_diff = conversion_info['size'] - original_info['size']
        size_percent = (size_diff / original_info['size']) * 100 if original_info['size'] > 0 else 0
        
        if size_diff < 0:
            print(f"  Size saved: {format_file_size(abs(size_diff))} ({abs(size_percent):.1f}% reduction)")
        elif size_diff > 0:
            print(f"  Size increased: {format_file_size(size_diff)} ({size_percent:.1f}% increase)")
        else:
            print(f"  Size unchanged")
                
        # User prompt
        print("\n" + "-" * 80)
        if not user_choice:
            print("What would you like to do?")
            print("  [o] Keep only ORIGINAL file (delete converted)")
            print("  [c] Keep only CONVERTED file (delete original)")
            print("  [b] Keep BOTH files")
            print("  [S] Keep SMALLER file for ALL remaining files")
            print("  [O] Keep only ORIGINAL file for ALL remaining files")
            print("  [C] Keep only CONVERTED file for ALL remaining files")
            print("  [q] Quit review (keep all remaining files)")
        
        while True:
            try:
                choice = ''
                if not user_choice:
                    choice = input("\nYour choice [o/c/b/S/O/C/q]: ").strip()

                    if choice == 'O':
                        user_choice = 'keep_original'
                    elif choice == 'C':
                        user_choice = 'keep_converted'
                    elif choice == 'S':
                        user_choice = 'keep_smaller'
                    elif choice == 'q':
                        user_choice = 'keep_both'
                
                if choice.lower() == 'c' or user_choice == 'keep_converted':
                    delete_file(original_path, "original file")
                    bytes_saved_total += original_info['size'] - conversion_info['size']
                    break
                elif choice.lower() == 'o' or user_choice == 'keep_original':
                    delete_file(converted_path, "converted file")
                    bytes_saved_total += conversion_info['size'] - original_info['size']
                    break
                elif choice.lower() == 's' or user_choice == 'keep_smaller':
                    if conversion_info['size'] <= original_info['size']:
                        delete_file(original_path, "original file, which is larger")
                        bytes_saved_total += original_info['size'] - conversion_info['size']
                    else:
                        delete_file(converted_path, "converted file, which is larger")
                        bytes_saved_total += conversion_info['size'] - original_info['size']
                    break
                elif choice.lower() == 'b' or user_choice == 'keep_both':
                    print(f"✓ Keeping both files")
                    bytes_saved_total -= conversion_info['size']
                    break
                elif choice.lower() == 'q':
                    print(f"\n✓ Exiting review. All remaining files are kept.")
                    break
                else:
                    print("Invalid choice. Please enter 'o', 'c', 'b', 'S', 'O', 'C', or 'q'.")
            except EOFError:
                print("\nError: Input ended. Keeping all remaining files.")
                return
            except KeyboardInterrupt:
                print("\nError: Review interrupted. Keeping all remaining files.")
                return
        
        print()
    
    print("=" * 80)
    print("REVIEW COMPLETE")
    if bytes_saved_total > 0:
        print(f"\nTotal disk space saved: {format_file_size(bytes_saved_total)}")
    else:
        print(f"\nAdditional disk space used: {format_file_size(-bytes_saved_total)}")
    print("=" * 80)


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
        '-c',
        '--convert',
        action='store_true',
        help='Convert qualified video files to h265 in MKV container after listing'
    )
    
    parser.add_argument(
        '-o',
        '--output-dir',
        type=str,
        metavar='DIR',
        help='Output directory for converted files (default: same directory as input file)'
    )

    parser.add_argument(
        '--skip-review',
        action='store_true',
        help='Skip review after conversion (default: false)'
    )
    
    parser.add_argument(
        '--codec',
        choices=['h264', 'h265'],
        default='h265',
        help='Video codec to use for conversion: h264 or h265 (default: h265)'
    )
    
    parser.add_argument(
        '--preset',
        choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'],
        default='slow',
        help='Encoding preset (speed/quality tradeoff). Options: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow (default: slow)'
    )
    
    parser.add_argument(
        '--crf',
        type=int,
        default=23,
        metavar='VALUE',
        help='Constant Rate Factor for quality (1-51, lower is better quality, default: 23)'
    )
    
    parser.add_argument(
        'directory',
        type=str,
        help='Directory to scan for video files'
    )

    # Validate CRF value
    args = parser.parse_args()
    if args.crf < 1 or args.crf > 51:
        parser.error('CRF value must be between 1 and 51')
    
    return args


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
    
    # Get video info and sort (done once for both listing and conversion)
    video_data = get_and_sort_video_data(video_files, sort_by=args.sort_by, reverse=args.reverse)
    
    # Print results
    print_video_list(video_data, directory, sort_by=args.sort_by, reverse=args.reverse)
    
    # Exit if no conversion requested
    if not args.convert:
        return
    
    # Validate output directory if provided
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.exists():
            print(f"\nCreating output directory: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)
        elif not output_dir.is_dir():
            print(f"Error: Output path '{args.output_dir}' exists but is not a directory.", 
                    file=sys.stderr)
            sys.exit(1)
    
    # Process conversions
    conversion_result = process_conversions(video_data, output_dir, args.codec, args.preset, args.crf)

    if args.skip_review:
        print("\nSkipping review.")
        return
    
    # Review converted files
    review_results(conversion_result)


if __name__ == '__main__':
    main()
