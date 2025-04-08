import os
from datetime import datetime

def compile_txt_files():
    # Define the input and output paths
    input_dir = os.path.join(os.path.dirname(__file__), 'output')
    output_file = os.path.join(os.path.dirname(__file__), 'documentation.txt')

    # Get current timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Open the output file in write mode
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write(f"=== Compilation generated on {timestamp} ===\n\n")

        for filename in sorted(os.listdir(input_dir)):
            if filename.endswith('.txt'):
                file_path = os.path.join(input_dir, filename)

                # Write header with decorative separator
                outfile.write("=" * 80 + "\n")
                outfile.write(f"📄 File: {filename}\n")
                outfile.write("=" * 80 + "\n\n")

                with open(file_path, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())
                    outfile.write("\n\n")

        outfile.write("=" * 80 + "\n")
        outfile.write("✅ End of compilation\n")

    print(f"All .txt files compiled into {output_file}")

# Only run when executed directly, not when imported
if __name__ == "__main__":
    compile_txt_files()
