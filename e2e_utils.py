import datetime
import inspect
import traceback

import torch


class E2EModel(torch.nn.Module):
    def __init__(self, model_id, AutoModelClass, AutoProcessorClass, post_process_function_name):
        super().__init__()
        self.model = AutoModelClass.from_pretrained(model_id)
        self.processor = AutoProcessorClass.from_pretrained(model_id, use_fast=True)
        self.post_process_function = getattr(self.processor, post_process_function_name)
        self.signature_params = inspect.signature(self.post_process_function).parameters

    def get_tensors_inputs(self, images, **preprocess_kwargs):
        images, pre_pre_processed_args, pre_pre_processed_kwargs = self.processor.preprocess(
            images=images, intermediate_return=True, return_tensors="pt", **preprocess_kwargs
        )
        self.pre_pre_processed_args = pre_pre_processed_args
        self.pre_pre_processed_kwargs = pre_pre_processed_kwargs

        return torch.stack(images)

    def forward(self, images, post_process_kwargs):
        preprocessed_inputs = self.processor._preprocess(
            images, *self.pre_pre_processed_args, **self.pre_pre_processed_kwargs
        )
        outputs = self.model(**preprocessed_inputs)
        outputs = self.post_process_function(outputs, **post_process_kwargs)

        return outputs


def format_detailed_error(exc, context=""):
    """
    Format detailed error information including file paths and line numbers
    """
    error_info = {"error_type": type(exc).__name__, "error_message": str(exc), "traceback_details": []}

    # Get the full traceback
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)

    print(f"\n🔍 DETAILED ERROR ANALYSIS {context}")
    print("=" * 60)
    print(f"❌ Error Type: {error_info['error_type']}")
    print(f"💬 Error Message: {error_info['error_message']}")

    print("\n📍 FULL TRACEBACK:")
    print("-" * 40)

    # Parse each line of the traceback for file/line info
    for i, line in enumerate(tb):
        if line.strip():
            print(f"   {line.rstrip()}")

            # Extract file and line number information
            if 'File "' in line and ", line " in line:
                try:
                    file_part = line.split('File "')[1].split('", line ')[0]
                    line_part = line.split(", line ")[1].split(",")[0]
                    error_info["traceback_details"].append({"file": file_part, "line": int(line_part)})
                except:
                    pass

    # Highlight the most relevant error location
    if error_info["traceback_details"]:
        last_location = error_info["traceback_details"][-1]
        print("\n🎯 PRIMARY ERROR LOCATION:")
        print(f"   📁 File: {last_location['file']}")
        print(f"   📍 Line: {last_location['line']}")

        # Try to show context around the error line
        try:
            with open(last_location["file"], "r") as f:
                lines = f.readlines()
                error_line = last_location["line"] - 1  # 0-indexed

                print(f"\n📖 CODE CONTEXT (lines {max(1, error_line - 2)}-{min(len(lines), error_line + 3)}):")
                print("-" * 40)

                for i in range(max(0, error_line - 2), min(len(lines), error_line + 3)):
                    marker = ">>>" if i == error_line else "   "
                    print(f"{marker} {i + 1:3d}| {lines[i].rstrip()}")

        except Exception as file_read_error:
            print(f"   ⚠️  Could not read source context: {file_read_error}")

    print("=" * 60)
    return error_info


def save_error_report(error_info, model_name, step_name):
    """
    Save detailed error report to a file for later analysis
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_name.replace("/", "_")
    filename = f"error_report_{safe_model_name}_{step_name}_{timestamp}.txt"

    try:
        with open(filename, "w") as f:
            f.write("ERROR REPORT\n")
            f.write("============\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Step: {step_name}\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            f.write(f"Error Type: {error_info['error_type']}\n")
            f.write(f"Error Message: {error_info['error_message']}\n\n")
            f.write("Traceback Details:\n")
            for detail in error_info["traceback_details"]:
                f.write(f"  File: {detail['file']}, Line: {detail['line']}\n")

        print(f"   💾 Error report saved to: {filename}")
        return filename
    except Exception as e:
        print(f"   ⚠️  Could not save error report: {e}")
        return None
