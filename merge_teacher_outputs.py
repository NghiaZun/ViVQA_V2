"""
merge_teacher_outputs.py
Merge multiple JSONL files và loại bỏ duplicates theo img_id
Author: Nghia-Duong
"""

import os
import json
from collections import OrderedDict
import argparse

def merge_jsonl_files(input_files, output_file, verbose=True):
    """
    Merge nhiều JSONL files, loại bỏ duplicates theo (img_id, question) pair
    Ưu tiên: File đầu tiên trong list có priority cao nhất
    
    Args:
        input_files: List các file paths cần merge
        output_file: Output file path
        verbose: In thông tin chi tiết
    """
    
    # Dùng OrderedDict để giữ thứ tự và loại duplicate
    # KEY: (img_id, question) để support multiple questions per image
    merged_data = OrderedDict()
    stats = {
        'total_lines': 0,
        'valid_entries': 0,
        'duplicates': 0,
        'errors': 0
    }
    
    print(f"[INFO] 🔄 Starting merge process...")
    print(f"[INFO] Number of input files: {len(input_files)}")
    
    for file_idx, input_file in enumerate(input_files, 1):
        if not os.path.exists(input_file):
            print(f"[WARN] ⚠️  File not found, skipping: {input_file}")
            continue
            
        print(f"\n[INFO] Processing file {file_idx}/{len(input_files)}: {input_file}")
        file_lines = 0
        file_added = 0
        file_skipped = 0
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                stats['total_lines'] += 1
                file_lines += 1
                
                try:
                    data = json.loads(line.strip())
                    img_id = str(data.get('img_id', '')).strip()
                    question = str(data.get('question', '')).strip()
                    
                    if not img_id:
                        stats['errors'] += 1
                        if verbose:
                            print(f"  [WARN] Line {line_num}: Missing img_id")
                        continue
                    
                    if not question:
                        stats['errors'] += 1
                        if verbose:
                            print(f"  [WARN] Line {line_num}: Missing question")
                        continue
                    
                    # KEY: (img_id, question) pair để support multiple questions per image
                    key = (img_id, question)
                    
                    # Nếu (img_id, question) chưa tồn tại thì thêm vào
                    if key not in merged_data:
                        merged_data[key] = data
                        stats['valid_entries'] += 1
                        file_added += 1
                    else:
                        stats['duplicates'] += 1
                        file_skipped += 1
                        
                except json.JSONDecodeError as e:
                    stats['errors'] += 1
                    if verbose:
                        print(f"  [ERROR] Line {line_num}: JSON decode error - {e}")
                except Exception as e:
                    stats['errors'] += 1
                    if verbose:
                        print(f"  [ERROR] Line {line_num}: {e}")
        
        print(f"  ✓ Lines read: {file_lines}")
        print(f"  ✓ New entries added: {file_added}")
        print(f"  ✓ Duplicates skipped: {file_skipped}")
    
    # Ghi ra file output
    print(f"\n[INFO] 💾 Writing merged data to: {output_file}")
    
    # Backup file cũ nếu tồn tại
    if os.path.exists(output_file):
        backup_file = output_file + ".backup"
        print(f"[INFO] Backing up existing file to: {backup_file}")
        os.rename(output_file, backup_file)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for key, data in merged_data.items():
            # key is (img_id, question) tuple, data is the full entry
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    # In thống kê
    print(f"\n{'='*60}")
    print(f"[INFO] ✅ Merge completed successfully!")
    print(f"{'='*60}")
    print(f"Total lines processed:     {stats['total_lines']:,}")
    print(f"Valid entries written:     {stats['valid_entries']:,}")
    print(f"Duplicates removed:        {stats['duplicates']:,}")
    print(f"Errors encountered:        {stats['errors']:,}")
    print(f"Output file:               {output_file}")
    print(f"Output file size:          {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    print(f"{'='*60}")
    
    return stats

def verify_jsonl_file(file_path):
    """Kiểm tra tính hợp lệ của JSONL file"""
    print(f"\n[INFO] 🔍 Verifying file: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found!")
        return False
    
    total_lines = 0
    unique_ids = set()
    unique_pairs = set()
    duplicate_pairs = 0
    errors = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            try:
                data = json.loads(line.strip())
                img_id = str(data.get('img_id', '')).strip()
                question = str(data.get('question', '')).strip()
                
                if img_id:
                    unique_ids.add(img_id)
                    
                    if question:
                        pair = (img_id, question)
                        if pair in unique_pairs:
                            duplicate_pairs += 1
                        else:
                            unique_pairs.add(pair)
                        
            except Exception as e:
                errors += 1
                if errors <= 5:  # Chỉ in 5 errors đầu
                    print(f"  [ERROR] Line {line_num}: {e}")
    
    print(f"  Total lines:           {total_lines:,}")
    print(f"  Unique img_ids:        {len(unique_ids):,}")
    print(f"  Unique (img_id, q):    {len(unique_pairs):,}")
    print(f"  Duplicate pairs:       {duplicate_pairs:,}")
    print(f"  Errors:                {errors:,}")
    print(f"  Avg questions/image:   {len(unique_pairs)/len(unique_ids) if unique_ids else 0:.2f}")
    
    if duplicate_pairs > 0:
        print(f"  [WARN] ⚠️  File contains {duplicate_pairs} duplicate (img_id, question) pairs!")
    if errors > 0:
        print(f"  [WARN] ⚠️  File contains {errors} invalid lines!")
    
    return duplicate_pairs == 0 and errors == 0

def auto_merge_before_resume(checkpoint_path, output_path):
    """
    Tự động merge checkpoint và output file nếu cả 2 đều tồn tại
    Trả về path của file đã merged để resume
    """
    if not os.path.exists(checkpoint_path):
        return output_path if os.path.exists(output_path) else None
    
    if not os.path.exists(output_path):
        return checkpoint_path
    
    # Cả 2 đều tồn tại - cần merge
    print(f"\n[INFO] 🔄 Both checkpoint and output exist - auto-merging...")
    merged_path = output_path + ".merged"
    
    stats = merge_jsonl_files(
        input_files=[checkpoint_path, output_path],
        output_file=merged_path,
        verbose=False
    )
    
    # Backup output cũ và replace bằng merged
    backup_path = output_path + ".pre_merge_backup"
    print(f"[INFO] Backing up {output_path} to {backup_path}")
    os.rename(output_path, backup_path)
    os.rename(merged_path, output_path)
    
    print(f"[INFO] ✅ Auto-merge completed: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(
        description='Merge JSONL files và loại bỏ duplicates',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge 2 files
  python merge_teacher_outputs.py file1.jsonl file2.jsonl -o merged.jsonl
  
  # Merge với verification
  python merge_teacher_outputs.py file1.jsonl file2.jsonl -o merged.jsonl --verify
  
  # Auto-merge checkpoint trước khi resume
  python merge_teacher_outputs.py --auto-merge
  
  # Kaggle paths
  python merge_teacher_outputs.py \\
    /kaggle/input/teacher-2-12/teacher_outputs_gt_guided.jsonl \\
    /kaggle/working/teacher_outputs_gt_guided.jsonl \\
    -o /kaggle/working/teacher_outputs_merged.jsonl
        """
    )
    
    parser.add_argument('input_files', nargs='*', help='Input JSONL files to merge')
    parser.add_argument('-o', '--output', help='Output merged JSONL file')
    parser.add_argument('--verify', action='store_true', help='Verify output file after merge')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    parser.add_argument('--auto-merge', action='store_true', help='Auto-merge checkpoint + output for resume')
    parser.add_argument('--checkpoint', default='/kaggle/input/teacher-2-12/teacher_outputs_gt_guided.jsonl',
                       help='Checkpoint file path for auto-merge')
    parser.add_argument('--working', default='/kaggle/working/teacher_outputs_gt_guided.jsonl',
                       help='Working file path for auto-merge')
    
    args = parser.parse_args()
    
    # Auto-merge mode
    if args.auto_merge:
        result = auto_merge_before_resume(args.checkpoint, args.working)
        if result:
            print(f"\n[INFO] ✅ Ready to resume from: {result}")
            if args.verify:
                verify_jsonl_file(result)
        else:
            print(f"[ERROR] No files found to merge!")
        return
    
    # Manual merge mode
    if not args.input_files or not args.output:
        print("[ERROR] Please provide input files and output path, or use --auto-merge")
        parser.print_help()
        return
    
    # Merge files
    stats = merge_jsonl_files(
        input_files=args.input_files,
        output_file=args.output,
        verbose=not args.quiet
    )
    
    # Verify if requested
    if args.verify:
        is_valid = verify_jsonl_file(args.output)
        if is_valid:
            print(f"\n[INFO] ✅ Verification passed!")
        else:
            print(f"\n[WARN] ⚠️  Verification found issues!")

if __name__ == "__main__":
    # Nếu không có arguments, dùng default paths cho Kaggle
    import sys
    if len(sys.argv) == 1:
        print("[INFO] Using default Kaggle paths...")
        
        default_inputs = [
            "/kaggle/input/teacher-2-12/teacher_outputs_gt_guided.jsonl",
            "/kaggle/working/teacher_outputs_gt_guided.jsonl"
        ]
        default_output = "/kaggle/working/teacher_outputs_merged.jsonl"
        
        # Filter existing files
        existing_files = [f for f in default_inputs if os.path.exists(f)]
        
        if not existing_files:
            print("[ERROR] No input files found! Please specify input files.")
            sys.exit(1)
        
        print(f"[INFO] Input files: {existing_files}")
        print(f"[INFO] Output file: {default_output}")
        
        stats = merge_jsonl_files(existing_files, default_output)
        verify_jsonl_file(default_output)
    else:
        main()
