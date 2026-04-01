#!/usr/bin/env python3
"""
Post-Implementation Cleanup Script
Run this after Phase 1 improvements to finalize the setup
"""

import os
import shutil
import sys

def main():
    print("=" * 60)
    print("Intelligent PDF Search - Phase 1 Cleanup")
    print("=" * 60)
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    modules_dir = os.path.join(project_root, "modules")
    
    # Step 1: Backup old answer_builder.py
    old_file = os.path.join(modules_dir, "answer_builder.py")
    backup_file = os.path.join(modules_dir, "answer_builder.py.backup")
    
    if os.path.exists(old_file):
        print("\n✅ Step 1: Backing up old answer_builder.py...")
        shutil.copy2(old_file, backup_file)
        print(f"   → Backup saved to: {backup_file}")
        os.remove(old_file)
        print(f"   → Old file removed")
    else:
        print("\n⚠️  Step 1: Old answer_builder.py not found (already cleaned?)")
    
    # Step 2: Rename new file
    new_file = os.path.join(modules_dir, "answer_builder_new.py")
    final_file = os.path.join(modules_dir, "answer_builder.py")
    
    if os.path.exists(new_file):
        print("\n✅ Step 2: Moving answer_builder_new.py to answer_builder.py...")
        shutil.move(new_file, final_file)
        print(f"   → File renamed successfully")
    else:
        print("\n❌ Step 2: answer_builder_new.py not found!")
        return False
    
    # Step 3: Update app.py import
    app_file = os.path.join(project_root, "app.py")
    print("\n✅ Step 3: Updating app.py import...")
    
    with open(app_file, 'r') as f:
        content = f.read()
    
    old_import = "from modules.answer_builder_new import build_answer"
    new_import = "from modules.answer_builder import build_answer"
    
    if old_import in content:
        content = content.replace(old_import, new_import)
        with open(app_file, 'w') as f:
            f.write(content)
        print(f"   → Import updated in app.py")
    else:
        print(f"   ⚠️  Import not found or already updated")
    
    # Step 4: Create logs directory
    logs_dir = os.path.join(project_root, "logs")
    print("\n✅ Step 4: Creating logs directory...")
    os.makedirs(logs_dir, exist_ok=True)
    print(f"   → Logs directory ready at: {logs_dir}")
    
    # Step 5: Verify all critical files exist
    print("\n✅ Step 5: Verifying critical files...")
    required_files = [
        "app.py",
        "modules/config.py",
        "modules/search_engine.py",
        "modules/vectorizer.py",
        "modules/answer_builder.py",
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = os.path.join(project_root, file_path)
        if os.path.exists(full_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - MISSING!")
            all_exist = False
    
    if not all_exist:
        print("\n❌ Some files are missing! Cleanup may have failed.")
        return False
    
    # Step 6: Test imports
    print("\n✅ Step 6: Testing imports...")
    try:
        sys.path.insert(0, project_root)
        from modules import config
        from modules import search_engine
        from modules import vectorizer
        from modules import answer_builder
        print("   ✅ All modules import successfully")
    except Exception as e:
        print(f"   ❌ Import error: {e}")
        return False
    
    # Final summary
    print("\n" + "=" * 60)
    print("✅ PHASE 1 CLEANUP COMPLETE!")
    print("=" * 60)
    print("\nYour project is now ready with:")
    print("  • Centralized configuration (modules/config.py)")
    print("  • Improved error handling throughout")
    print("  • Comprehensive logging setup")
    print("  • Production-ready code structure")
    print("\nNext steps:")
    print("  1. Start the app:  python app.py")
    print("  2. Upload a PDF and test search functionality")
    print("  3. Check logs in the 'logs' directory for debug info")
    print("\nFor Phase 2 improvements (database, caching, async):")
    print("  See IMPROVEMENTS_SUMMARY.md for recommendations")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
