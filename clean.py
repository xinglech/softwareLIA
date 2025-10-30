#!/usr/bin/env python3
"""
Cleanup script for Software Lock-In Amplifier output files
Removes all generated log files and plot images from the current directory
"""

import os
import glob
import argparse
import sys
from datetime import datetime

def find_output_files():
    """Find all LIA output files in current directory"""
    patterns = [
        'lockin_analysis_*.log',
        'lockin_analysis_*.png',
        'lockin_analysis_*.jpg',
        'lockin_analysis_*.pdf',
        '7panel_comprehensive_analysis.png',
        '7panel_comprehensive_analysis.pdf',
        '8panel_comprehensive_analysis.png',
        '8panel_comprehensive_analysis.pdf',
        'xyrt_components.png',
        'comprehensive_performance_analysis.png',
        'comprehensive_performance_analysis.pdf'
    ]
    
    files_to_delete = []
    for pattern in patterns:
        files_to_delete.extend(glob.glob(pattern))
    
    return files_to_delete

def clean_output_files(dry_run=False, confirm=True):
    """Clean up all output files"""
    files_to_delete = find_output_files()
    
    if not files_to_delete:
        print("No Lock-In Amplifier output files found in current directory.")
        return
    
    print("Found the following Lock-In Amplifier output files:")
    for i, file in enumerate(files_to_delete, 1):
        file_size = os.path.getsize(file) if os.path.exists(file) else 0
        print(f"  {i:2d}. {file} ({file_size/1024:.1f} KB)")
    
    total_size = sum(os.path.getsize(f) for f in files_to_delete if os.path.exists(f))
    print(f"\nTotal files: {len(files_to_delete)}")
    print(f"Total size: {total_size/1024:.1f} KB")
    
    if dry_run:
        print("\nDRY RUN: No files will be deleted.")
        return
    
    if confirm and len(files_to_delete) > 0:
        response = input(f"\nAre you sure you want to delete {len(files_to_delete)} files? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Cleanup cancelled.")
            return
    
    # Delete files
    deleted_count = 0
    deleted_size = 0
    
    for file in files_to_delete:
        try:
            if os.path.exists(file):
                file_size = os.path.getsize(file)
                os.remove(file)
                print(f"Deleted: {file}")
                deleted_count += 1
                deleted_size += file_size
        except Exception as e:
            print(f"Error deleting {file}: {e}")
    
    print(f"\nCleanup completed:")
    print(f"  Files deleted: {deleted_count}")
    print(f"  Space freed: {deleted_size/1024:.1f} KB")

def main():
    """Main command-line interface"""
    parser = argparse.ArgumentParser(
        description='Cleanup script for Software Lock-In Amplifier output files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python clean.py                    # Interactive cleanup with confirmation
  python clean.py --force            # Force delete without confirmation
  python clean.py --dry-run          # Show what would be deleted without actually deleting
  python clean.py --yes              # Auto-confirm deletion
        """
    )
    
    parser.add_argument('-f', '--force', action='store_true',
                       help='Force deletion without confirmation')
    parser.add_argument('-d', '--dry-run', action='store_true',
                       help='Show what would be deleted without actually deleting')
    parser.add_argument('-y', '--yes', action='store_true',
                       help='Auto-confirm deletion (non-interactive)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    print("Software Lock-In Amplifier - Cleanup Utility")
    print("=" * 50)
    print(f"Current directory: {os.getcwd()}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        clean_output_files(
            dry_run=args.dry_run,
            confirm=not (args.force or args.yes)
        )
    except Exception as e:
        print(f"Error during cleanup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()