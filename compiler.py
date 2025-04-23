# compiler.py

import os
from datetime import datetime
import traceback

# --- Import Configuration ---
import config # Import hardcoded config values

# --- Configuration from config.py ---
# Use directories defined in config.py as defaults
DEFAULT_INPUT_DIR = config.OUTPUT_DIR
DEFAULT_COMPILED_OUTPUT_FILE = config.DEFAULT_COMPILED_OUTPUT_FILE

def compile_txt_files(input_dir: str = DEFAULT_INPUT_DIR, output_file: str = DEFAULT_COMPILED_OUTPUT_FILE):
    """
    Compiles all .txt files from the input directory into a single output file.
    Uses default directories from config.py if not specified.

    Args:
        input_dir: The directory containing the .txt files to compile.
                   Defaults to config.OUTPUT_DIR.
        output_file: The path to the final compiled documentation file.
                     Defaults to config.DEFAULT_COMPILED_OUTPUT_FILE.
    """
    print(f"[Compiler] Starting compilation from: {os.path.abspath(input_dir)}")
    print(f"[Compiler] Output will be saved to: {os.path.abspath(output_file)}")

    if not os.path.isdir(input_dir):
        print(f"[Compiler] Error: Input directory '{os.path.abspath(input_dir)}' not found or is not a directory.")
        return

    # Ensure the directory for the output *file* exists
    # (e.g., if output_file is specified as 'some_new_dir/compiled.txt')
    output_dir_for_file = os.path.dirname(output_file)
    if output_dir_for_file and not os.path.exists(output_dir_for_file):
        try:
            print(f"[Compiler] Creating output directory: {os.path.abspath(output_dir_for_file)}")
            os.makedirs(output_dir_for_file)
        except OSError as e:
            print(f"[Compiler] Error: Could not create output directory '{os.path.abspath(output_dir_for_file)}': {e}")
            traceback.print_exc()
            return
        except Exception as e:
             print(f"[Compiler] Error: An unexpected error occurred creating output directory: {type(e).__name__}")
             traceback.print_exc()
             return


    # Get list of .txt files, handling potential errors
    try:
        # List files and filter for .txt extension
        all_files = os.listdir(input_dir)
        txt_files = sorted([f for f in all_files if f.endswith('.txt') and os.path.isfile(os.path.join(input_dir, f))])
    except OSError as e:
        print(f"[Compiler] Error: Could not read input directory '{os.path.abspath(input_dir)}': {e}")
        traceback.print_exc()
        return
    except Exception as e:
        print(f"[Compiler] Error: An unexpected error occurred listing files: {type(e).__name__}")
        traceback.print_exc()
        return


    if not txt_files:
        print(f"[Compiler] No .txt files found in '{os.path.abspath(input_dir)}'. Nothing to compile.")
        return

    print(f"[Compiler] Found {len(txt_files)} .txt files to compile.")

    # Get current timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z') # Added timezone info if available

    try:
        # Open the output file in write mode with UTF-8 encoding
        with open(output_file, 'w', encoding='utf-8') as outfile:
            # Write header information
            outfile.write(f"=== Compilation Generated: {timestamp} ===\n")
            outfile.write(f"=== Source Directory: {os.path.abspath(input_dir)} ===\n\n")

            for filename in txt_files:
                file_path = os.path.join(input_dir, filename)

                try:
                    # Write separator and file header
                    outfile.write("=" * 80 + "\n")
                    outfile.write(f"📄 Source File: {filename}\n")
                    outfile.write("=" * 80 + "\n\n")

                    # Open and read the source file
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                        content = infile.read()
                        outfile.write(content)
                        outfile.write("\n\n") # Add spacing after file content

                except OSError as e:
                    print(f"[Compiler] Warning: Could not read file '{filename}': {e}. Skipping.")
                    # Write error message into the compiled file
                    outfile.write(f"[Compiler Error: Could not read file '{filename}']\n\n")
                except Exception as e:
                     print(f"[Compiler] Warning: Unexpected error reading file '{filename}': {type(e).__name__}. Skipping.")
                     outfile.write(f"[Compiler Error: Unexpected error reading file '{filename}']\n\n")


            # Write footer
            outfile.write("=" * 80 + "\n")
            outfile.write("✅ End of Compilation\n")
            outfile.write("=" * 80 + "\n")

        print(f"[Compiler] Compilation successful. Output saved to: {os.path.abspath(output_file)}")

    except OSError as e:
        print(f"[Compiler] Error: Could not write to output file '{os.path.abspath(output_file)}': {e}")
        traceback.print_exc()
    except Exception as e:
         print(f"[Compiler] Error: An unexpected error occurred during compilation: {type(e).__name__}")
         traceback.print_exc()

# Only run when executed directly, not when imported
if __name__ == "__main__":
    print("Running compiler directly...")
    # Calls the function with defaults taken from config.py
    compile_txt_files()
    print("Compiler finished.")