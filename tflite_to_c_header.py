import argparse
import os

def convert_tflite_to_header(tflite_path, header_path, var_name="g_model"):
    with open(tflite_path, 'rb') as f:
        data = f.read()

    with open(header_path, 'w') as f:
        f.write(f"#ifndef {var_name.upper()}_H\n")
        f.write(f"#define {var_name.upper()}_H\n\n")
        f.write(f"// TFLite Model: {tflite_path}\n")
        f.write(f"// Size: {len(data)} bytes\n")
        f.write(f"const unsigned char {var_name}[] = {{\n")
        
        chunk_size = 12
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i+chunk_size]
            hex_chunk = ", ".join(f"0x{b:02x}" for b in chunk)
            if i + chunk_size < len(data):
                f.write(f"  {hex_chunk},\n")
            else:
                f.write(f"  {hex_chunk}\n")
                
        f.write("};\n\n")
        f.write(f"const unsigned int {var_name}_len = {len(data)};\n\n")
        f.write(f"#endif // {var_name.upper()}_H\n")
    
    print(f"Converted {tflite_path} to {header_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TFLite model to C header")
    parser.add_argument("--input", default="face_model_quantized.tflite", help="Input .tflite file")
    parser.add_argument("--output", default="model_data.h", help="Output .h file")
    parser.add_argument("--var", default="g_model", help="C variable name")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
    else:
        convert_tflite_to_header(args.input, args.output, args.var)
